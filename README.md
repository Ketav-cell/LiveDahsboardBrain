# BFI Monitoring Dashboard

Real-time Brain Fragility Index (BFI) monitoring dashboard simulation with a professional ICU-style interface.

## Features

- **Live BFI Gauge** — Large circular arc gauge with color-coded zones (Green/Yellow/Red)
- **Scrolling Trend Graph** — 5-minute rolling window of BFI values at 2 Hz
- **EEG Feature Panels** — Real-time display of Alpha-band power, Theta/Alpha ratio, Coherence, Entropy, and Instability variance
- **Session Summary** — Stop monitoring to see full session statistics including zone-time percentages, min/max/mean BFI, and a complete session recording graph
- **Realistic Simulation** — Gaussian noise with occasional drift trends and mean-reversion dynamics

## Zone Definitions

| Zone     | BFI Range | Meaning                       |
|----------|-----------|-------------------------------|
| Green    | 0–39      | Normal / Stable               |
| Yellow   | 40–69     | Caution / Elevated fragility  |
| Red      | 70–100    | Critical / High fragility     |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python bfi_dashboard.py
```

1. Click **START** to begin the monitoring simulation
2. Observe the BFI gauge, trend graph, and EEG feature values update in real time
3. Click **STOP** to freeze values and view full session summary statistics
4. Click **RESET** to clear data and start a new session

## Technical Details

- **Update rate:** 2 Hz (500 ms interval)
- **Graph window:** 300 data points (5 minutes)
- **GUI framework:** PyQt5 with matplotlib for charting
- **Simulation:** Gaussian noise + random drift trends + mean-reversion to healthy baseline (~50)
