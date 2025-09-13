# app.py
import os
import logging
from datetime import datetime, timezone, timedelta

from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename

from image_utils import extract_ohlc_from_image
from signal_engine import compute_indicators_and_signal, SIGNAL_FIELDS, calibrate_confidence

import pandas as pd
import json
import uuid

# Setup
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
LOG_FOLDER = "logs"
os.makedirs(LOG_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("signal_app")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# In-memory signal log for demo (replace with DB in production)
SIGNAL_LOG = []

# Timezone UTC+6
UTC6 = timezone(timedelta(hours=6))

# Small whitelist of assets + suggested trading times & risk guidance (tunable)
ASSET_GUIDANCE = {
    # times are rough guidance windows in UTC+6 (HH:MM) — general advice only
    "USD/BDT OTC": {"best": "09:00-16:00", "avoid": "00:00-05:00", "risk_pct": 0.5},
    "USCrude OTC": {"best": "10:00-18:00", "avoid": "22:00-04:00", "risk_pct": 0.5},
    "UKBrent OTC": {"best": "10:00-18:00", "avoid": "00:00-05:00", "risk_pct": 0.6},
    "Gold OTC": {"best": "08:30-16:30", "avoid": "23:00-03:00", "risk_pct": 0.5},
    "USD/INR OTC": {"best": "10:00-17:00", "avoid": "00:00-05:00", "risk_pct": 0.4},
    # generic fallback
    "DEFAULT": {"best": "09:00-17:00", "avoid": "00:00-05:00", "risk_pct": 0.5},
}

@app.route("/")
def index():
    return render_template("index.html", assets=list(ASSET_GUIDANCE.keys()))

@app.route("/upload", methods=["POST"])
def upload():
    """
    Expects multipart/form-data:
      - file: image
      - asset: asset name (string)
      - timezone_offset_minutes: optional (client timezone offset debug)
    Returns JSON with signal metadata on success.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "no file uploaded"}), 400
        file = request.files["file"]
        asset = request.form.get("asset", "DEFAULT")
        fname = secure_filename(file.filename or f"img_{uuid.uuid4().hex}.png")
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        file.save(saved_path)
        logger.info("Saved upload: %s for asset %s", saved_path, asset)

        # Extract OHLC (best-effort)
        try:
            df = extract_ohlc_from_image(saved_path, debug=False)
        except Exception as e:
            logger.exception("Image->OHLC failed")
            return jsonify({"error": "image_processing_failed", "detail": str(e)}), 500

        if df is None or len(df) < 10:
            return jsonify({"error": "insufficient_candles", "detail": "need >=10 candles"}), 400

        # Compute indicators and aggregated signal
        result = compute_indicators_and_signal(df)
        # Calibrate confidence using historical and image quality heuristics
        result["confidence"] = calibrate_confidence(result, df)

        # Execution time in UTC+6 (server)
        now_utc6 = datetime.now(timezone.utc).astimezone(UTC6)
        result["time_utc6"] = now_utc6.strftime("%Y-%m-%d %H:%M:%S %Z")
        result["asset"] = asset
        result["image"] = saved_path
        result["id"] = uuid.uuid4().hex

        # attach guidance
        guidance = ASSET_GUIDANCE.get(asset, ASSET_GUIDANCE["DEFAULT"])
        result["guidance"] = guidance

        # Push to in-memory log and emit realtime event
        SIGNAL_LOG.append(result)
        # keep last 500
        if len(SIGNAL_LOG) > 500:
            SIGNAL_LOG.pop(0)

        socketio.emit("new_signal", result)
        return jsonify(result)

    except Exception as e:
        logger.exception("Unexpected error in /upload")
        return jsonify({"error": "server_error", "detail": str(e)}), 500

@app.route("/signals")
def signals():
    n = int(request.args.get("n", 50))
    return jsonify(SIGNAL_LOG[-n:])

@app.route("/asset-guidance")
def asset_guidance():
    return jsonify(ASSET_GUIDANCE)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    # eventlet recommended; debug False for production
    socketio.run(app, host="0.0.0.0", port=5000)