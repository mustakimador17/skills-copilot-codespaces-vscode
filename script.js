// static/js/main.js
document.addEventListener("DOMContentLoaded", async () => {
  const socket = io();
  const feed = document.getElementById("feed");
  const sendBtn = document.getElementById("send");
  const fileInput = document.getElementById("file");
  const status = document.getElementById("status");
  const assetSelect = document.getElementById("asset");
  const guidanceDiv = document.getElementById("guidance");
  const clearBtn = document.getElementById("clear");

  // load assets and guidance from server
  const ag = await fetch("/asset-guidance").then(r => r.json());
  const assets = Object.keys(ag);
  assetSelect.innerHTML = assets.map(a => `<option value="${a}">${a}</option>`).join("");
  guidanceDiv.innerHTML = assets.map(a => `<div><b>${a}</b> — Best: ${ag[a].best} | Avoid: ${ag[a].avoid} | Risk%: ${ag[a].risk_pct}</div>`).join("");

  socket.on("new_signal", (data) => {
    addToFeed(data);
  });

  function addToFeed(s) {
    const el = document.createElement("div");
    el.className = "feed-item";
    const left = document.createElement("div");
    left.className = "feed-left";
    const right = document.createElement("div");
    right.className = "feed-right";

    left.innerHTML = `<div><strong>${s.asset}</strong> <span class="muted">• ${s.time_utc6}</span></div>
                      <div style="font-size:13px;color:#374151">${s.logic}</div>`;
    const badgeClass = s.direction === "Up" ? "badge up" : "badge down";
    right.innerHTML = `<div class="${badgeClass}">${s.direction}</div><div style="font-size:12px">${s.confidence}%</div>`;

    el.appendChild(left);
    el.appendChild(right);
    feed.insertBefore(el, feed.firstChild);
  }

  sendBtn.addEventListener("click", async () => {
    if (!fileInput.files || !fileInput.files[0]) {
      alert("Please select an image");
      return;
    }
    sendBtn.disabled = true;
    status.textContent = "Uploading…";
    const fd = new FormData();
    fd.append("file", fileInput.files[0]);
    fd.append("asset", assetSelect.value);
    try {
      const resp = await fetch("/upload", { method: "POST", body: fd });
      const j = await resp.json();
      if (!resp.ok) {
        status.textContent = "Error: " + (j && (j.error || j.detail) ? (j.error || j.detail) : "Unknown");
      } else {
        addToFeed(j);
        status.textContent = "Signal generated";
      }
    } catch (e) {
      status.textContent = "Network error";
    } finally {
      sendBtn.disabled = false;
      // keep device awake on mobile? we avoid forcing anything
    }
  });

  clearBtn.addEventListener("click", () => {
    feed.innerHTML = "";
  });

});