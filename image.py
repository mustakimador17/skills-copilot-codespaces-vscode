# image_utils.py
"""
Improved image->OHLC extraction pipeline with:
- multi-method detection: color-based + edge-based
- axis detection via Hough lines
- OCR fallback for price labels
- validation and sanity checks, returns pandas.DataFrame
"""
import cv2
import numpy as np
import pytesseract
import pandas as pd
from datetime import datetime, timedelta
import logging
import math

logger = logging.getLogger("image_utils")
logger.setLevel(logging.INFO)

def extract_ohlc_from_image(path, debug=False):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image unreadable")

    h, w = img.shape[:2]
    # normalization step: if large, downscale while preserving aspect ratio for speed
    max_dim = 1600
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # attempt to find chart rectangular area by locating longest horizontal/vertical lines via Canny+Hough
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=int(w * 0.3), maxLineGap=20)
    # default crop area (fallback)
    chart_x1, chart_x2 = int(w * 0.06), int(w * 0.96)
    chart_y1, chart_y2 = int(h * 0.08), int(h * 0.82)
    if lines is not None:
        # try to find top/bottom by scanning horizontal lines with large length
        ys = []
        xs = []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            if abs(y2 - y1) < 5 and abs(x2 - x1) > w * 0.4:
                ys.append(y1)
            if abs(x2 - x1) < 5 and abs(y2 - y1) > h * 0.05:
                xs.append(x1)
        if ys:
            topy, boty = min(ys), max(ys)
            # add padding
            chart_y1 = max(0, topy - 6)
            chart_y2 = min(h, boty + 12)
        if xs:
            leftx = min(xs)
            chart_x1 = max(0, leftx + 2)
            chart_x2 = min(w - 1, int(w * 0.96))
    chart = img[chart_y1:chart_y2, chart_x1:chart_x2]
    ch_h, ch_w = chart.shape[:2]
    if ch_h < 30 or ch_w < 60:
        raise ValueError("Chart region too small, image likely not a chart or cropping failed")

    # attempt color-based candle detection
    hsv = cv2.cvtColor(chart, cv2.COLOR_BGR2HSV)
    # adaptive color thresholds: try to detect dominant candle color pairs
    # find two dominant colors via kmeans on hue channel
    hue = cv2.cvtColor(chart, cv2.COLOR_BGR2HSV)[:, :, 0].reshape(-1, 1).astype(np.float32)
    try:
        import cv2 as _cv
        _, labels, centers = _cv.kmeans(hue, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 3, _cv.KMEANS_PP_CENTERS)
        centers = centers.flatten()
        centers.sort()
    except Exception:
        centers = [0, 60, 120]

    # heuristic masks for green/red using approx hue centers
    # common green ~ 40-80, red ~ 0 or >160 -> we'll try wide ranges
    lower_green = np.array([30, 30, 20]); upper_green = np.array([100, 255, 255])
    lower_red1 = np.array([0, 30, 20]); upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 30, 20]); upper_red2 = np.array([179, 255, 255])
    mask_g = cv2.inRange(hsv, lower_green, upper_green)
    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_r = cv2.bitwise_or(mask_r1, mask_r2)
    mask = cv2.bitwise_or(mask_g, mask_r)

    # if color mask too sparse, fallback to edge-based detection (candles as vertical rectangles)
    if cv2.countNonZero(mask) < (ch_h * ch_w) * 0.01:
        # use morphological operations on grayscale to find vertical strokes
        grad = cv2.Sobel(cv2.cvtColor(chart, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=3)
        grad = np.absolute(grad).astype(np.uint8)
        _, thresh = cv2.threshold(grad, 30, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        # find contours
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cand_boxes = []
        for c in cnts:
            x, y, wbox, hbox = cv2.boundingRect(c)
            if hbox > ch_h * 0.06 and wbox < ch_w * 0.2:
                cand_boxes.append((x, y, wbox, hbox))
    else:
        # use mask contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        mm = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cand_boxes = []
        for c in cnts:
            x, y, wbox, hbox = cv2.boundingRect(c)
            if hbox > ch_h * 0.04 and wbox < ch_w * 0.18:
                cand_boxes.append((x, y, wbox, hbox))

    if not cand_boxes:
        raise ValueError("No candle-like shapes detected; try different chart style or provide samples")

    cand_boxes = sorted(cand_boxes, key=lambda r: r[0])
    # merge overlapping boxes horizontally (to handle thin contours)
    merged = []
    for rect in cand_boxes:
        if not merged:
            merged.append(rect)
            continue
        x,y,ww,hh = rect
        mx,my,mw,mh = merged[-1]
        if x <= mx + mw * 0.4:
            # merge
            nx = min(x, mx)
            nw = max(mx + mw, x + ww) - nx
            ny = min(y, my)
            nh = max(my + mh, y + hh) - ny
            merged[-1] = (nx, ny, nw, nh)
        else:
            merged.append(rect)
    cand_boxes = merged

    # Limit to most recent 200 candles to avoid extreme memory/time
    cand_boxes = cand_boxes[-200:]

    # Map pixel Y to price via OCR on left axis when available
    left_axis = img[chart_y1:chart_y2, max(0, chart_x1 - int(w * 0.12)):chart_x1]
    try:
        ocr_cfg = "--psm 6 -c tessedit_char_whitelist=0123456789.-"
        text = pytesseract.image_to_string(left_axis, config=ocr_cfg)
        import re
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        prices = [float(x) for x in nums] if nums else []
    except Exception:
        prices = []

    # if we have at least two price labels, estimate linear mapping between pixel Y and price
    if len(prices) >= 2:
        # find approximate vertical positions of price labels by scanning left_axis for text bounding boxes
        try:
            boxes = pytesseract.image_to_data(left_axis, config=ocr_cfg, output_type=pytesseract.Output.DICT)
            coords = []
            for i, txt in enumerate(boxes["text"]):
                t = txt.strip()
                if not t:
                    continue
                try:
                    val = float(t)
                except Exception:
                    continue
                # bounding box center
                cy = boxes["top"][i] + boxes["height"][i] // 2
                coords.append((val, cy))
            # if we got coords, map top pixel (min cy) -> max price etc.
            if coords:
                vals, cys = zip(*coords)
                max_price = max(vals); min_price = min(vals)
                top_pix = min(cys); bottom_pix = max(cys)
                def pixel_to_price(y):
                    # y is relative to chart top (0..ch_h)
                    # map (top_pix -> max_price) and (bottom_pix -> min_price)
                    rel = (y - top_pix) / (bottom_pix - top_pix) if bottom_pix != top_pix else (y / ch_h)
                    return max_price - rel * (max_price - min_price)
            else:
                raise Exception("OCR coords empty")
        except Exception:
            # fallback mapping: use first and last price numeric reading positions approx
            max_price = max(prices); min_price = min(prices)
            def pixel_to_price(y):
                return max_price - (y / ch_h) * (max_price - min_price)
    else:
        # fallback mapping: try to read top and bottom price from chart title or right axis via OCR
        # worst-case: return relative prices 0..1 scaled to arbitrary
        last_close_guess = 1000.0  # placeholder
        def pixel_to_price(y):
            # returns a relative price scaled so distances make sense; calling code must check validity
            return (1.0 - float(y) / ch_h) * 1000.0

    # For each cand box, estimate OHLC by sampling vertical color distribution
    candles = []
    for (x, y, ww, hh) in cand_boxes:
        rect = chart[y:y + hh, x:x + ww]
        # compute column-wise presence to approximate wick extents
        rect_gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
        # detect non-background pixels
        nz = np.where(rect_gray < 250)  # darker pixels
        if nz[0].size == 0:
            continue
        top_local = int(nz[0].min())
        bottom_local = int(nz[0].max())
        # body detection: find central vertical slice
        mid_col = rect[:, max(0, ww // 2 - 2): min(ww, ww // 2 + 3)]
        mid_gray = cv2.cvtColor(mid_col, cv2.COLOR_BGR2GRAY)
        nz_body = np.where(mid_gray < 250)
        if nz_body[0].size == 0:
            body_top_local = top_local
            body_bottom_local = bottom_local
        else:
            body_top_local = int(nz_body[0].min())
            body_bottom_local = int(nz_body[0].max())
        # sample color to decide bullish/bearish: average hue in body
        hsv_body = cv2.cvtColor(mid_col, cv2.COLOR_BGR2HSV)
        mean_hue = float(np.mean(hsv_body[:, :, 0])) if hsv_body.size else 0.0
        # compute prices
        wick_top = y + top_local
        wick_bottom = y + bottom_local
        body_top = y + body_top_local
        body_bottom = y + body_bottom_local
        high = pixel_to_price(wick_top)
        low = pixel_to_price(wick_bottom)
        # bullish if mean_hue in green-ish range
        if 30 < mean_hue < 100:
            close_p = pixel_to_price(body_top)
            open_p = pixel_to_price(body_bottom)
        else:
            open_p = pixel_to_price(body_top)
            close_p = pixel_to_price(body_bottom)
        # sanity checks
        if not (math.isfinite(open_p) and math.isfinite(high) and math.isfinite(low) and math.isfinite(close_p)):
            continue
        # ensure high >= max(open,close) and low <= min(open,close)
        high = max(high, open_p, close_p)
        low = min(low, open_p, close_p)
        candles.append({"x": x, "open": open_p, "high": high, "low": low, "close": close_p})

    # Build DF. Map candles to chronological order left->right
    candles = sorted(candles, key=lambda c: c["x"])
    if not candles:
        raise ValueError("No candles after processing")

    # assign timestamps assuming uniform 1-minute spacing ending at now UTC
    n = len(candles)
    now = datetime.utcnow().replace(second=0, microsecond=0)
    times = [now - timedelta(minutes=(n - 1 - i)) for i in range(n)]
    rows = []
    for t, c in zip(times, candles):
        rows.append({"time": t, "open": c["open"], "high": c["high"], "low": c["low"], "close": c["close"]})
    df = pd.DataFrame(rows).set_index("time")
    # basic sanity: remove extreme values or NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty or len(df) < 5:
        raise ValueError("Insufficient valid candles extracted")
    return df