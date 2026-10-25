/* Market Regime Predictor - Terminal Client Application */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const tickerInput = document.getElementById("ticker-input");
  const periodSelect = document.getElementById("period-select");
  const regimesSelect = document.getElementById("regimes-select");
  const executeBtn = document.getElementById("execute-btn");
  const presetBtns = document.querySelectorAll(".preset-btn");
  const clearConsoleBtn = document.getElementById("clear-console-btn");
  const consoleOutput = document.getElementById("console-output");

  const clockDisplay = document.getElementById("current-clock");
  const activeRegimeName = document.getElementById("active-regime-name");
  const activeRegimeConfidence = document.getElementById("active-regime-confidence");
  const spotPrice = document.getElementById("spot-price");
  const spotDelta = document.getElementById("spot-delta");
  const rsiVal = document.getElementById("rsi-val");
  const rsiState = document.getElementById("rsi-state");
  const volVal = document.getElementById("vol-val");
  const stratSharpe = document.getElementById("strat-sharpe");
  const benchSharpe = document.getElementById("bench-sharpe");
  const maxDd = document.getElementById("max-dd");
  const benchDd = document.getElementById("bench-dd");

  const regimeTableBody = document.getElementById("regime-table-body");
  const transitionTableBody = document.getElementById("transition-table-body");
  const featureList = document.getElementById("feature-list");

  const priceCanvas = document.getElementById("price-chart-canvas");
  const equityCanvas = document.getElementById("equity-chart-canvas");

  // Realtime Clock
  function updateClock() {
    const now = new Date();
    clockDisplay.textContent = now.toISOString().slice(11, 19) + " UTC";
  }
  setInterval(updateClock, 1000);
  updateClock();

  // Logger helper
  function log(message, type = "info") {
    const now = new Date().toISOString().slice(11, 19);
    const entry = document.createElement("div");
    entry.className = `log-entry ${type}`;
    entry.textContent = `[${now}] ${message}`;
    consoleOutput.appendChild(entry);
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
  }

  clearConsoleBtn.addEventListener("click", () => {
    consoleOutput.innerHTML = "";
    log("Console buffer flushed.", "system");
  });

  // Preset Buttons
  presetBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      presetBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      tickerInput.value = btn.dataset.ticker;
      triggerAnalysis();
    });
  });

  executeBtn.addEventListener("click", () => {
    triggerAnalysis();
  });

  tickerInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      triggerAnalysis();
    }
  });

  // Canvas resize helper
  function resizeCanvas(canvas) {
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    return { width: rect.width, height: rect.height, ctx };
  }

  // Draw Price Chart
  function drawPriceChart(priceData) {
    if (!priceData || priceData.length === 0) return;
    const { width, height, ctx } = resizeCanvas(priceCanvas);

    ctx.clearRect(0, 0, width, height);

    const padLeft = 50;
    const padRight = 20;
    const padTop = 20;
    const padBottom = 30;

    const chartW = width - padLeft - padRight;
    const chartH = height - padTop - padBottom;

    const prices = priceData.map((d) => d.close);
    const minP = Math.min(...prices) * 0.98;
    const maxP = Math.max(...prices) * 1.02;

    // Draw Grid Lines
    ctx.strokeStyle = "#1c2129";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = padTop + (chartH / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(width - padRight, y);
      ctx.stroke();

      const priceVal = maxP - ((maxP - minP) / 5) * i;
      ctx.fillStyle = "#484f58";
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText(priceVal.toFixed(1), padLeft - 6, y + 3);
    }

    const n = priceData.length;
    const xStep = chartW / (n - 1);

    // Draw Regime Background Shading
    for (let i = 0; i < n - 1; i++) {
      const x1 = padLeft + i * xStep;
      const x2 = padLeft + (i + 1) * xStep;
      const color = priceData[i].regime_color || "#30363d";

      ctx.fillStyle = color;
      ctx.globalAlpha = 0.08;
      ctx.fillRect(x1, padTop, x2 - x1, chartH);
      ctx.globalAlpha = 1.0;
    }

    // Draw SMA 50
    ctx.strokeStyle = "#58a6ff";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    priceData.forEach((d, i) => {
      const x = padLeft + i * xStep;
      const y = padTop + chartH - ((d.sma_50 - minP) / (maxP - minP)) * chartH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw Price Line with Segmented Colors
    for (let i = 0; i < n - 1; i++) {
      const x1 = padLeft + i * xStep;
      const y1 = padTop + chartH - ((priceData[i].close - minP) / (maxP - minP)) * chartH;
      const x2 = padLeft + (i + 1) * xStep;
      const y2 = padTop + chartH - ((priceData[i + 1].close - minP) / (maxP - minP)) * chartH;

      ctx.strokeStyle = priceData[i].regime_color || "#00ff66";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }

    // Time Axis
    ctx.fillStyle = "#8b949e";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    const stepCount = 5;
    for (let i = 0; i <= stepCount; i++) {
      const idx = Math.floor(((n - 1) / stepCount) * i);
      const x = padLeft + idx * xStep;
      const dateStr = priceData[idx].date.slice(5);
      ctx.fillText(dateStr, x, height - 10);
    }
  }

  // Draw Equity Curve Chart
  function drawEquityChart(equityData) {
    if (!equityData || equityData.length === 0) return;
    const { width, height, ctx } = resizeCanvas(equityCanvas);

    ctx.clearRect(0, 0, width, height);

    const padLeft = 60;
    const padRight = 20;
    const padTop = 15;
    const padBottom = 25;

    const chartW = width - padLeft - padRight;
    const chartH = height - padTop - padBottom;

    const stratVals = equityData.map((d) => d.strategy_equity);
    const benchVals = equityData.map((d) => d.benchmark_equity);
    const minVal = Math.min(...stratVals, ...benchVals) * 0.95;
    const maxVal = Math.max(...stratVals, ...benchVals) * 1.05;

    // Grid
    ctx.strokeStyle = "#1c2129";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 3; i++) {
      const y = padTop + (chartH / 3) * i;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(width - padRight, y);
      ctx.stroke();

      const val = maxVal - ((maxVal - minVal) / 3) * i;
      ctx.fillStyle = "#484f58";
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText("$" + (val / 1000).toFixed(0) + "k", padLeft - 6, y + 3);
    }

    const n = equityData.length;
    const xStep = chartW / (n - 1);

    // Benchmark Curve (Gray)
    ctx.strokeStyle = "#8b949e";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    equityData.forEach((d, i) => {
      const x = padLeft + i * xStep;
      const y = padTop + chartH - ((d.benchmark_equity - minVal) / (maxVal - minVal)) * chartH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Strategy Curve (Emerald)
    ctx.strokeStyle = "#00ff66";
    ctx.lineWidth = 2;
    ctx.beginPath();
    equityData.forEach((d, i) => {
      const x = padLeft + i * xStep;
      const y = padTop + chartH - ((d.strategy_equity - minVal) / (maxVal - minVal)) * chartH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Legend
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#00ff66";
    ctx.fillText("— REGIME ALLOCATION", padLeft + 10, padTop + 15);
    ctx.fillStyle = "#8b949e";
    ctx.fillText("— BUY & HOLD BENCHMARK", padLeft + 160, padTop + 15);
  }

  // Populate Tables and UI
  function updateUI(data) {
    const cur = data.current_regime;
    const tel = data.latest_telemetry;
    const bm = data.backtest.benchmark_metrics;
    const sm = data.backtest.strategy_metrics;

    // Active Regime Card
    activeRegimeName.textContent = cur.name.toUpperCase();
    activeRegimeName.style.color = cur.color;
    activeRegimeConfidence.textContent = `CONFIDENCE: ${cur.confidence_pct}% [P=${cur.probabilities[cur.regime_id]}]`;

    // Spot Price Card
    spotPrice.textContent = `$${tel.close.toFixed(2)}`;
    const sign = tel.change >= 0 ? "+" : "";
    spotDelta.textContent = `DELTA: ${sign}${tel.change.toFixed(2)} (${sign}${tel.change_pct.toFixed(2)}%)`;
    spotDelta.style.color = tel.change >= 0 ? "#00ff66" : "#ff3344";

    // RSI
    rsiVal.textContent = tel.rsi_14.toFixed(1);
    rsiState.textContent = tel.rsi_14 > 70 ? "OVERBOUGHT" : tel.rsi_14 < 30 ? "OVERSOLD" : "NEUTRAL_MOMENTUM";

    // Vol
    volVal.textContent = `${tel.rolling_vol_20.toFixed(2)}%`;

    // Sharpe
    stratSharpe.textContent = sm.sharpe_ratio.toFixed(2);
    benchSharpe.textContent = `BENCHMARK: ${bm.sharpe_ratio.toFixed(2)}`;

    // Drawdown
    maxDd.textContent = `${sm.max_drawdown_pct.toFixed(2)}%`;
    benchDd.textContent = `BENCHMARK: ${bm.max_drawdown_pct.toFixed(2)}%`;

    // Regime Summary Table
    regimeTableBody.innerHTML = "";
    data.regime_summary.forEach((r) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="color: ${r.color}; font-weight: 600;">${r.name}</td>
        <td>${r.percentage.toFixed(1)}%</td>
        <td style="color: ${r.annualized_return_pct >= 0 ? "#00ff66" : "#ff3344"}">${r.annualized_return_pct.toFixed(1)}%</td>
        <td>${r.annualized_volatility_pct.toFixed(1)}%</td>
        <td>${r.sharpe_ratio.toFixed(2)}</td>
      `;
      regimeTableBody.appendChild(tr);
    });

    // Transition Matrix Table
    transitionTableBody.innerHTML = "";
    data.transition_matrix.forEach((row, i) => {
      const tr = document.createElement("tr");
      let rowHtml = `<td style="color: #8b949e">S${i}</td>`;
      row.forEach((prob) => {
        const intensity = Math.min(1, prob * 1.2);
        const colVal = `rgba(0, 255, 102, ${Math.max(0.2, intensity)})`;
        rowHtml += `<td style="color: ${prob > 0.4 ? colVal : '#8b949e'}">${prob.toFixed(2)}</td>`;
      });
      tr.innerHTML = rowHtml;
      transitionTableBody.appendChild(tr);
    });

    // Feature Importances
    featureList.innerHTML = "";
    const sortedFeatures = Object.entries(data.feature_importances).sort((a, b) => b[1] - a[1]);
    const maxImp = Math.max(...sortedFeatures.map((f) => f[1]), 0.01);
    sortedFeatures.forEach(([feat, imp]) => {
      const pct = (imp / maxImp) * 100;
      const item = document.createElement("div");
      item.className = "feature-bar-item";
      item.innerHTML = `
        <div class="feature-bar-header">
          <span>${feat}</span>
          <span>${(imp * 100).toFixed(1)}%</span>
        </div>
        <div class="feature-track">
          <div class="feature-fill" style="width: ${pct}%"></div>
        </div>
      `;
      featureList.appendChild(item);
    });

    // Render Canvases
    drawPriceChart(data.price_history);
    drawEquityChart(data.backtest.equity_timeseries);
  }

  // API Trigger
  async function triggerAnalysis() {
    const ticker = tickerInput.value.trim().toUpperCase() || "SPY";
    const period = periodSelect.value;
    const nRegimes = parseInt(regimesSelect.value, 10);

    executeBtn.disabled = true;
    executeBtn.textContent = "COMPUTING...";
    log(`Initiating quantitative inference for ${ticker} [Window: ${period}, Regimes: ${nRegimes}]...`, "info");

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, period, n_regimes: nRegimes }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Server inference error");
      }

      const data = await res.json();
      updateUI(data);

      log(`Analysis completed: ${data.sample_count} samples processed. Active Regime: ${data.current_regime.name}.`, "success");
      log(`Backtest: Strategy CAGR ${data.backtest.strategy_metrics.cagr_pct}%, Benchmark CAGR ${data.backtest.benchmark_metrics.cagr_pct}%.`, "info");
    } catch (err) {
      log(`Inference execution failed: ${err.message}`, "error");
    } finally {
      executeBtn.disabled = false;
      executeBtn.textContent = "EXECUTE_INFERENCE";
    }
  }

  window.addEventListener("resize", () => {
    // Redraw on window resize if data exists
    const ticker = tickerInput.value.trim();
    if (ticker) {
      triggerAnalysis();
    }
  });

  // Initial trigger
  triggerAnalysis();
});
