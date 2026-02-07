#!/usr/bin/env python3
"""
Brain Fragility Index (BFI) Real-Time Monitoring Dashboard
===========================================================
Professional ICU-style simulation dashboard for monitoring Brain Fragility Index.

Simulates real-time BFI data with realistic noise patterns and displays:
- Large BFI gauge meter with color-coded zones
- Scrolling 5-minute line graph
- Live EEG-derived feature values
- Session summary statistics on stop

Author: BFI Monitoring Systems
"""

import sys
import time
import threading
import numpy as np
from collections import deque
from datetime import datetime, timedelta

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QGridLayout, QSizePolicy, QSpacerItem
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QBrush, QLinearGradient, QPainterPath

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Color palette – ICU medical dark theme
# ---------------------------------------------------------------------------
COLORS = {
    'bg_dark':       '#0a0e17',
    'bg_panel':      '#111827',
    'bg_card':       '#1a2236',
    'border':        '#2a3a5c',
    'text_primary':  '#e0e6f0',
    'text_secondary':'#8892a8',
    'text_dim':      '#4a5568',
    'green':         '#00e676',
    'green_dark':    '#004d25',
    'yellow':        '#ffca28',
    'yellow_dark':   '#5c4800',
    'red':           '#ff1744',
    'red_dark':      '#5c0011',
    'accent_blue':   '#29b6f6',
    'accent_cyan':   '#00e5ff',
    'grid_line':     '#1e2d4a',
    'chart_bg':      '#0d1520',
}

ZONE_COLORS = {
    'green':  '#00e676',
    'yellow': '#ffca28',
    'red':    '#ff1744',
}


def bfi_zone_color(value):
    """Return the color string for a BFI value."""
    if value < 40:
        return COLORS['green']
    elif value < 70:
        return COLORS['yellow']
    return COLORS['red']


def bfi_zone_label(value):
    if value < 40:
        return 'NORMAL'
    elif value < 70:
        return 'CAUTION'
    return 'CRITICAL'


# ---------------------------------------------------------------------------
# Data Simulation Engine
# ---------------------------------------------------------------------------
class BFISimulator:
    """Generates realistic BFI data with Gaussian noise and occasional trends."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.bfi = 50.0
        self.trend = 0.0
        self.trend_duration = 0
        self._tick = 0

    def step(self):
        """Advance one time step (500 ms). Returns (bfi, features_dict)."""
        self._tick += 1

        # Occasionally start a new drift trend
        if self.trend_duration <= 0 and np.random.random() < 0.05:
            self.trend = np.random.uniform(-2.0, 2.0)
            self.trend_duration = np.random.randint(4, 20)

        if self.trend_duration > 0:
            self.bfi += self.trend * 0.3
            self.trend_duration -= 1
        else:
            self.trend = 0.0

        # Gaussian noise
        self.bfi += np.random.normal(0, 1.2)

        # Mean-reversion toward 50 (healthy center)
        self.bfi += (50.0 - self.bfi) * 0.02

        # Clamp
        self.bfi = float(np.clip(self.bfi, 0, 100))

        # Derive correlated feature values
        features = self._derive_features(self.bfi)
        return self.bfi, features

    @staticmethod
    def _derive_features(bfi):
        """Generate EEG features correlated with BFI."""
        noise = lambda s=0.02: np.random.normal(0, s)

        # Alpha-band power (µV²): higher BFI → lower alpha
        alpha = max(0.1, 12.0 - 0.08 * bfi + noise(0.4))

        # Theta/alpha ratio: higher BFI → higher ratio
        theta_alpha = max(0.1, 0.8 + 0.015 * bfi + noise(0.04))

        # Coherence (0-1): higher BFI → lower coherence
        coherence = float(np.clip(0.85 - 0.005 * bfi + noise(0.02), 0, 1))

        # Entropy: higher BFI → higher entropy
        entropy = float(np.clip(0.4 + 0.004 * bfi + noise(0.02), 0, 1))

        # Instability variance: higher BFI → higher variance
        instability = max(0.0, 0.01 + 0.0008 * bfi + noise(0.005))

        return {
            'alpha_power':  round(alpha, 2),
            'theta_alpha':  round(theta_alpha, 3),
            'coherence':    round(coherence, 3),
            'entropy':      round(entropy, 3),
            'instability':  round(instability, 4),
        }


# ---------------------------------------------------------------------------
# Custom Gauge Widget
# ---------------------------------------------------------------------------
class GaugeWidget(QWidget):
    """Circular arc gauge for BFI score display."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._target = 0.0
        self.setMinimumSize(280, 240)

    def set_value(self, v):
        self._target = float(np.clip(v, 0, 100))
        # Smooth interpolation
        self._value += (self._target - self._value) * 0.35
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        side = min(w, h)
        cx, cy = w / 2, h / 2 + 10

        radius = side * 0.42
        arc_width = side * 0.045

        # Draw background arc (full sweep)
        start_angle = 225 * 16  # degrees * 16 (Qt convention)
        span_angle = -270 * 16

        pen = QPen(QColor(COLORS['bg_card']), arc_width, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)
        rect_arc = self._arc_rect(cx, cy, radius)
        painter.drawArc(rect_arc, start_angle, span_angle)

        # Draw colored segments
        segments = [
            (0, 40, COLORS['green']),
            (40, 70, COLORS['yellow']),
            (70, 100, COLORS['red']),
        ]
        for lo, hi, color in segments:
            seg_start = 225 - (lo / 100) * 270
            seg_end = 225 - (hi / 100) * 270
            seg_span = (seg_end - seg_start) * 16
            pen_seg = QPen(QColor(color), arc_width * 0.4, Qt.SolidLine, Qt.RoundCap)
            pen_seg.setColor(QColor(color))
            painter.setPen(pen_seg)
            painter.drawArc(rect_arc, int(seg_start * 16), int(seg_span))

        # Draw active arc up to current value
        val_frac = self._value / 100.0
        active_span = -val_frac * 270 * 16
        color = bfi_zone_color(self._value)
        pen_active = QPen(QColor(color), arc_width, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen_active)
        painter.drawArc(rect_arc, start_angle, int(active_span))

        # Glow effect on the active arc
        glow_color = QColor(color)
        glow_color.setAlpha(40)
        pen_glow = QPen(glow_color, arc_width * 1.8, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen_glow)
        painter.drawArc(rect_arc, start_angle, int(active_span))

        # Draw needle
        needle_angle_deg = 225 - val_frac * 270
        needle_angle_rad = np.radians(needle_angle_deg)
        nx = cx + (radius * 0.75) * np.cos(needle_angle_rad)
        ny = cy - (radius * 0.75) * np.sin(needle_angle_rad)

        pen_needle = QPen(QColor('#ffffff'), 2.5, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen_needle)
        painter.drawLine(int(cx), int(cy), int(nx), int(ny))

        # Center dot
        painter.setBrush(QBrush(QColor(color)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(int(cx - 6), int(cy - 6), 12, 12)

        # BFI text
        font_big = QFont('Consolas', int(side * 0.14), QFont.Bold)
        painter.setFont(font_big)
        painter.setPen(QColor(color))
        bfi_text = f'{self._value:.1f}'
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(bfi_text)
        painter.drawText(int(cx - tw / 2), int(cy + radius * 0.45), bfi_text)

        # Zone label
        font_label = QFont('Consolas', int(side * 0.05))
        painter.setFont(font_label)
        zone_text = bfi_zone_label(self._value)
        tw2 = painter.fontMetrics().horizontalAdvance(zone_text)
        painter.drawText(int(cx - tw2 / 2), int(cy + radius * 0.6), zone_text)

        # Scale labels
        font_scale = QFont('Consolas', int(side * 0.035))
        painter.setFont(font_scale)
        painter.setPen(QColor(COLORS['text_dim']))
        for val in [0, 20, 40, 60, 80, 100]:
            a = np.radians(225 - (val / 100) * 270)
            lx = cx + (radius + arc_width * 1.2) * np.cos(a)
            ly = cy - (radius + arc_width * 1.2) * np.sin(a)
            t = str(val)
            tw3 = painter.fontMetrics().horizontalAdvance(t)
            painter.drawText(int(lx - tw3 / 2), int(ly + 5), t)

        painter.end()

    def _arc_rect(self, cx, cy, r):
        from PyQt5.QtCore import QRectF
        return QRectF(cx - r, cy - r, 2 * r, 2 * r)


# ---------------------------------------------------------------------------
# Real-time Chart (matplotlib)
# ---------------------------------------------------------------------------
class LiveChart(FigureCanvas):
    """Scrolling line chart embedded in Qt."""

    MAX_POINTS = 300  # 5 minutes at 2 Hz

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 2.6), dpi=100, facecolor=COLORS['chart_bg'])
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self._style_axes()

        self.data = deque(maxlen=self.MAX_POINTS)
        self.line, = self.ax.plot([], [], color=COLORS['accent_cyan'], linewidth=1.5, alpha=0.9)

        # Zone bands
        self.ax.axhspan(0, 40, facecolor=COLORS['green_dark'], alpha=0.15)
        self.ax.axhspan(40, 70, facecolor=COLORS['yellow_dark'], alpha=0.15)
        self.ax.axhspan(70, 100, facecolor=COLORS['red_dark'], alpha=0.15)

        # Zone boundary lines
        self.ax.axhline(40, color=COLORS['green'], linewidth=0.5, alpha=0.3)
        self.ax.axhline(70, color=COLORS['red'], linewidth=0.5, alpha=0.3)

        self.fig.tight_layout(pad=1.0)
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.15)

    def _style_axes(self):
        ax = self.ax
        ax.set_facecolor(COLORS['chart_bg'])
        ax.set_ylim(-2, 102)
        ax.set_ylabel('BFI', color=COLORS['text_secondary'], fontsize=9, fontfamily='monospace')
        ax.set_xlabel('Time (s)', color=COLORS['text_secondary'], fontsize=8, fontfamily='monospace')
        ax.tick_params(colors=COLORS['text_dim'], labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(COLORS['border'])
        ax.spines['left'].set_color(COLORS['border'])
        ax.grid(True, color=COLORS['grid_line'], linewidth=0.4, alpha=0.6)
        ax.set_title('BFI  TREND  (5 min window)', color=COLORS['text_secondary'],
                      fontsize=9, fontfamily='monospace', pad=6)

    def append(self, value):
        self.data.append(value)
        n = len(self.data)
        xs = np.arange(n) * 0.5  # 500 ms steps
        # Shift x so rightmost point = 0 s ago
        xs = xs - xs[-1]
        self.line.set_data(xs, list(self.data))
        self.ax.set_xlim(xs[0] - 1, xs[-1] + 1)

        # Color the line by latest zone
        self.line.set_color(bfi_zone_color(value))

        self.draw_idle()

    def show_full_session(self, all_data):
        """Replace scrolling view with full session data."""
        self.ax.clear()
        self._style_axes()
        self.ax.set_title('FULL  SESSION  RECORDING', color=COLORS['text_secondary'],
                          fontsize=9, fontfamily='monospace', pad=6)

        n = len(all_data)
        xs = np.arange(n) * 0.5
        ys = np.array(all_data)

        # Zone bands
        self.ax.axhspan(0, 40, facecolor=COLORS['green_dark'], alpha=0.15)
        self.ax.axhspan(40, 70, facecolor=COLORS['yellow_dark'], alpha=0.15)
        self.ax.axhspan(70, 100, facecolor=COLORS['red_dark'], alpha=0.15)
        self.ax.axhline(40, color=COLORS['green'], linewidth=0.5, alpha=0.3)
        self.ax.axhline(70, color=COLORS['red'], linewidth=0.5, alpha=0.3)

        # Plot with gradient-like coloring by segment
        for i in range(1, n):
            seg_color = bfi_zone_color(ys[i])
            self.ax.plot(xs[i-1:i+1], ys[i-1:i+1], color=seg_color, linewidth=1.2, alpha=0.85)

        # Mean line
        mean_val = np.mean(ys)
        self.ax.axhline(mean_val, color=COLORS['accent_blue'], linewidth=1, linestyle='--', alpha=0.6)
        self.ax.text(xs[-1] * 0.02, mean_val + 2, f'Mean: {mean_val:.1f}',
                     color=COLORS['accent_blue'], fontsize=8, fontfamily='monospace')

        self.ax.set_xlim(-1, xs[-1] + 1)
        self.ax.set_xlabel('Time (s)', color=COLORS['text_secondary'], fontsize=8, fontfamily='monospace')
        self.fig.tight_layout(pad=1.0)
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.15)
        self.draw()

    def reset_chart(self):
        self.data.clear()
        self.ax.clear()
        self._style_axes()
        self.line, = self.ax.plot([], [], color=COLORS['accent_cyan'], linewidth=1.5, alpha=0.9)
        self.ax.axhspan(0, 40, facecolor=COLORS['green_dark'], alpha=0.15)
        self.ax.axhspan(40, 70, facecolor=COLORS['yellow_dark'], alpha=0.15)
        self.ax.axhspan(70, 100, facecolor=COLORS['red_dark'], alpha=0.15)
        self.ax.axhline(40, color=COLORS['green'], linewidth=0.5, alpha=0.3)
        self.ax.axhline(70, color=COLORS['red'], linewidth=0.5, alpha=0.3)
        self.fig.tight_layout(pad=1.0)
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.15)
        self.draw()


# ---------------------------------------------------------------------------
# Feature Card Widget
# ---------------------------------------------------------------------------
class FeatureCard(QFrame):
    """Single feature value display card."""

    def __init__(self, label, unit, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 6px;
            }}
        """)
        self.setFixedHeight(80)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)

        self.label_widget = QLabel(label)
        self.label_widget.setFont(QFont('Consolas', 8))
        self.label_widget.setStyleSheet(f'color: {COLORS["text_secondary"]}; border: none;')

        self.value_widget = QLabel('--')
        self.value_widget.setFont(QFont('Consolas', 18, QFont.Bold))
        self.value_widget.setStyleSheet(f'color: {COLORS["accent_cyan"]}; border: none;')

        self.unit_widget = QLabel(unit)
        self.unit_widget.setFont(QFont('Consolas', 7))
        self.unit_widget.setStyleSheet(f'color: {COLORS["text_dim"]}; border: none;')

        layout.addWidget(self.label_widget)
        layout.addWidget(self.value_widget)
        layout.addWidget(self.unit_widget)

    def set_value(self, val_str, color=None):
        self.value_widget.setText(val_str)
        if color:
            self.value_widget.setStyleSheet(f'color: {color}; border: none;')


# ---------------------------------------------------------------------------
# Summary Panel
# ---------------------------------------------------------------------------
class SummaryPanel(QFrame):
    """Post-session summary statistics overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_panel']};
                border: 2px solid {COLORS['accent_blue']};
                border-radius: 10px;
            }}
        """)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 15, 20, 15)
        self.layout.setSpacing(6)
        self.hide()

    def populate(self, stats):
        # Clear existing
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        title = QLabel('SESSION  SUMMARY')
        title.setFont(QFont('Consolas', 14, QFont.Bold))
        title.setStyleSheet(f'color: {COLORS["accent_blue"]}; border: none;')
        title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f'color: {COLORS["border"]}; border: none; background: {COLORS["border"]};')
        sep.setFixedHeight(1)
        self.layout.addWidget(sep)

        rows = [
            ('Duration', stats['duration']),
            ('Samples', str(stats['samples'])),
            ('', ''),
            ('Average BFI', f"{stats['mean']:.1f}"),
            ('Minimum BFI', f"{stats['min']:.1f}"),
            ('Maximum BFI', f"{stats['max']:.1f}"),
            ('Std Deviation', f"{stats['std']:.2f}"),
            ('', ''),
            ('Green Zone  (0-39)', f"{stats['pct_green']:.1f}%"),
            ('Yellow Zone (40-69)', f"{stats['pct_yellow']:.1f}%"),
            ('Red Zone    (70-100)', f"{stats['pct_red']:.1f}%"),
            ('', ''),
            ('Final Alpha Power', f"{stats['final_features']['alpha_power']} µV²"),
            ('Final Theta/Alpha', f"{stats['final_features']['theta_alpha']}"),
            ('Final Coherence', f"{stats['final_features']['coherence']}"),
            ('Final Entropy', f"{stats['final_features']['entropy']}"),
            ('Final Instability', f"{stats['final_features']['instability']}"),
        ]

        for label, value in rows:
            if label == '' and value == '':
                spacer = QLabel('')
                spacer.setFixedHeight(4)
                spacer.setStyleSheet('border: none;')
                self.layout.addWidget(spacer)
                continue

            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFont(QFont('Consolas', 10))
            lbl.setStyleSheet(f'color: {COLORS["text_secondary"]}; border: none;')

            val = QLabel(value)
            val.setFont(QFont('Consolas', 10, QFont.Bold))

            # Color the zone percentages
            color = COLORS['text_primary']
            if 'Green' in label:
                color = COLORS['green']
            elif 'Yellow' in label:
                color = COLORS['yellow']
            elif 'Red' in label:
                color = COLORS['red']
            elif 'Average' in label or 'Mean' in label.lower():
                color = COLORS['accent_cyan']

            val.setStyleSheet(f'color: {color}; border: none;')
            val.setAlignment(Qt.AlignRight)

            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(val)
            self.layout.addLayout(row)

        self.show()


# ---------------------------------------------------------------------------
# Main Dashboard Window
# ---------------------------------------------------------------------------
class BFIDashboard(QMainWindow):
    """Main application window for BFI monitoring dashboard."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('BFI  Monitoring  Dashboard  —  Brain  Fragility  Index')
        self.setMinimumSize(1100, 720)
        self.resize(1200, 780)

        self.simulator = BFISimulator()
        self.running = False
        self.all_bfi_data = []
        self.last_features = {}
        self.session_start = None

        self._build_ui()
        self._setup_timer()

    # ---- UI Construction ----
    def _build_ui(self):
        central = QWidget()
        central.setStyleSheet(f'background-color: {COLORS["bg_dark"]};')
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 8, 12, 8)
        root_layout.setSpacing(8)

        # -- Header --
        header = self._build_header()
        root_layout.addLayout(header)

        # -- Body: left (gauge + features) | right (chart + summary) --
        body = QHBoxLayout()
        body.setSpacing(10)

        left_panel = self._build_left_panel()
        right_panel = self._build_right_panel()

        body.addLayout(left_panel, 35)
        body.addLayout(right_panel, 65)

        root_layout.addLayout(body, 1)

        # -- Footer status bar --
        self.status_label = QLabel('READY  —  Press START to begin monitoring')
        self.status_label.setFont(QFont('Consolas', 9))
        self.status_label.setStyleSheet(
            f'color: {COLORS["text_dim"]}; padding: 4px; '
            f'border-top: 1px solid {COLORS["border"]};'
        )
        root_layout.addWidget(self.status_label)

    def _build_header(self):
        layout = QHBoxLayout()

        # Title
        title = QLabel('BFI  MONITORING  SYSTEM')
        title.setFont(QFont('Consolas', 16, QFont.Bold))
        title.setStyleSheet(f'color: {COLORS["accent_blue"]};')

        subtitle = QLabel('Brain Fragility Index  ·  Real-Time Simulation')
        subtitle.setFont(QFont('Consolas', 9))
        subtitle.setStyleSheet(f'color: {COLORS["text_dim"]};')

        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        title_col.addWidget(title)
        title_col.addWidget(subtitle)

        layout.addLayout(title_col)
        layout.addStretch()

        # Clock
        self.clock_label = QLabel('')
        self.clock_label.setFont(QFont('Consolas', 11))
        self.clock_label.setStyleSheet(f'color: {COLORS["text_secondary"]};')
        layout.addWidget(self.clock_label)

        layout.addSpacing(20)

        # Elapsed
        self.elapsed_label = QLabel('ELAPSED  00:00')
        self.elapsed_label.setFont(QFont('Consolas', 11))
        self.elapsed_label.setStyleSheet(f'color: {COLORS["text_dim"]};')
        layout.addWidget(self.elapsed_label)

        layout.addSpacing(20)

        # Buttons
        btn_style_start = f"""
            QPushButton {{
                background-color: #004d25;
                color: {COLORS['green']};
                border: 1px solid {COLORS['green']};
                border-radius: 5px;
                padding: 8px 24px;
                font-family: Consolas;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #006633;
            }}
        """
        btn_style_stop = f"""
            QPushButton {{
                background-color: #5c0011;
                color: {COLORS['red']};
                border: 1px solid {COLORS['red']};
                border-radius: 5px;
                padding: 8px 24px;
                font-family: Consolas;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #800018;
            }}
        """
        btn_style_reset = f"""
            QPushButton {{
                background-color: #1a2236;
                color: {COLORS['accent_blue']};
                border: 1px solid {COLORS['accent_blue']};
                border-radius: 5px;
                padding: 8px 24px;
                font-family: Consolas;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #243050;
            }}
        """

        self.btn_start = QPushButton('▶  START')
        self.btn_start.setStyleSheet(btn_style_start)
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self._on_start)

        self.btn_stop = QPushButton('■  STOP')
        self.btn_stop.setStyleSheet(btn_style_stop)
        self.btn_stop.setCursor(Qt.PointingHandCursor)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setEnabled(False)

        self.btn_reset = QPushButton('⟳  RESET')
        self.btn_reset.setStyleSheet(btn_style_reset)
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_reset.hide()

        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_reset)

        return layout

    def _build_left_panel(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Gauge
        gauge_frame = QFrame()
        gauge_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_panel']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)
        gauge_layout = QVBoxLayout(gauge_frame)
        gauge_layout.setContentsMargins(8, 4, 8, 4)

        gauge_title = QLabel('BRAIN  FRAGILITY  INDEX')
        gauge_title.setFont(QFont('Consolas', 9, QFont.Bold))
        gauge_title.setStyleSheet(f'color: {COLORS["text_secondary"]}; border: none;')
        gauge_title.setAlignment(Qt.AlignCenter)
        gauge_layout.addWidget(gauge_title)

        self.gauge = GaugeWidget()
        gauge_layout.addWidget(self.gauge)

        layout.addWidget(gauge_frame)

        # Feature cards
        features_title = QLabel('EEG  FEATURES')
        features_title.setFont(QFont('Consolas', 9, QFont.Bold))
        features_title.setStyleSheet(f'color: {COLORS["text_secondary"]};')
        layout.addWidget(features_title)

        self.card_alpha = FeatureCard('ALPHA-BAND POWER', 'µV²')
        self.card_theta = FeatureCard('THETA / ALPHA RATIO', 'ratio')
        self.card_coherence = FeatureCard('COHERENCE', 'index (0-1)')
        self.card_entropy = FeatureCard('ENTROPY', 'index (0-1)')
        self.card_instability = FeatureCard('INSTABILITY VARIANCE', 'σ²')

        for card in [self.card_alpha, self.card_theta, self.card_coherence,
                     self.card_entropy, self.card_instability]:
            layout.addWidget(card)

        layout.addStretch()
        return layout

    def _build_right_panel(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Chart container
        chart_frame = QFrame()
        chart_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_panel']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)
        chart_layout = QVBoxLayout(chart_frame)
        chart_layout.setContentsMargins(6, 6, 6, 6)

        self.chart = LiveChart()
        chart_layout.addWidget(self.chart)

        layout.addWidget(chart_frame, 55)

        # Summary panel (hidden initially)
        self.summary_panel = SummaryPanel()
        layout.addWidget(self.summary_panel, 45)

        # Zone legend
        legend_layout = QHBoxLayout()
        legend_layout.addStretch()
        for zone_name, lo, hi, color in [
            ('NORMAL', 0, 39, COLORS['green']),
            ('CAUTION', 40, 69, COLORS['yellow']),
            ('CRITICAL', 70, 100, COLORS['red']),
        ]:
            dot = QLabel('●')
            dot.setFont(QFont('Consolas', 10))
            dot.setStyleSheet(f'color: {color};')
            lbl = QLabel(f'{zone_name} ({lo}-{hi})')
            lbl.setFont(QFont('Consolas', 8))
            lbl.setStyleSheet(f'color: {COLORS["text_dim"]};')
            legend_layout.addWidget(dot)
            legend_layout.addWidget(lbl)
            legend_layout.addSpacing(16)
        legend_layout.addStretch()
        layout.addLayout(legend_layout)

        return layout

    # ---- Timer ----
    def _setup_timer(self):
        self.timer = QTimer(self)
        self.timer.setInterval(500)  # 500 ms = 2 Hz
        self.timer.timeout.connect(self._tick)

        # Clock update timer
        self.clock_timer = QTimer(self)
        self.clock_timer.setInterval(1000)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start()
        self._update_clock()

    def _update_clock(self):
        now = datetime.now().strftime('%H:%M:%S')
        self.clock_label.setText(now)

        if self.running and self.session_start:
            elapsed = datetime.now() - self.session_start
            mins = int(elapsed.total_seconds()) // 60
            secs = int(elapsed.total_seconds()) % 60
            self.elapsed_label.setText(f'ELAPSED  {mins:02d}:{secs:02d}')
            self.elapsed_label.setStyleSheet(f'color: {COLORS["accent_cyan"]};')

    # ---- Actions ----
    def _on_start(self):
        self.running = True
        self.session_start = datetime.now()
        self.all_bfi_data.clear()
        self.simulator.reset()
        self.chart.reset_chart()
        self.summary_panel.hide()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_reset.hide()
        self.status_label.setText('● MONITORING ACTIVE')
        self.status_label.setStyleSheet(
            f'color: {COLORS["green"]}; padding: 4px; '
            f'border-top: 1px solid {COLORS["border"]};'
        )
        self.timer.start()

    def _on_stop(self):
        self.running = False
        self.timer.stop()
        self.btn_stop.setEnabled(False)
        self.btn_reset.show()

        self.status_label.setText('■ MONITORING STOPPED — Session complete')
        self.status_label.setStyleSheet(
            f'color: {COLORS["red"]}; padding: 4px; '
            f'border-top: 1px solid {COLORS["border"]};'
        )

        # Calculate summary stats
        data = np.array(self.all_bfi_data)
        if len(data) == 0:
            return

        elapsed = datetime.now() - self.session_start
        mins = int(elapsed.total_seconds()) // 60
        secs = int(elapsed.total_seconds()) % 60

        n_green = np.sum(data < 40)
        n_yellow = np.sum((data >= 40) & (data < 70))
        n_red = np.sum(data >= 70)
        total = len(data)

        stats = {
            'duration': f'{mins:02d}:{secs:02d}',
            'samples': total,
            'mean': float(np.mean(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'std': float(np.std(data)),
            'pct_green': 100.0 * n_green / total,
            'pct_yellow': 100.0 * n_yellow / total,
            'pct_red': 100.0 * n_red / total,
            'final_features': self.last_features,
        }

        self.summary_panel.populate(stats)
        self.chart.show_full_session(self.all_bfi_data)

    def _on_reset(self):
        self.all_bfi_data.clear()
        self.simulator.reset()
        self.chart.reset_chart()
        self.summary_panel.hide()
        self.gauge.set_value(0)
        self.btn_start.setEnabled(True)
        self.btn_reset.hide()
        self.elapsed_label.setText('ELAPSED  00:00')
        self.elapsed_label.setStyleSheet(f'color: {COLORS["text_dim"]};')

        for card in [self.card_alpha, self.card_theta, self.card_coherence,
                     self.card_entropy, self.card_instability]:
            card.set_value('--', COLORS['accent_cyan'])

        self.status_label.setText('READY  —  Press START to begin monitoring')
        self.status_label.setStyleSheet(
            f'color: {COLORS["text_dim"]}; padding: 4px; '
            f'border-top: 1px solid {COLORS["border"]};'
        )

    # ---- Simulation tick ----
    def _tick(self):
        if not self.running:
            return

        bfi, features = self.simulator.step()
        self.all_bfi_data.append(bfi)
        self.last_features = features

        # Update gauge
        self.gauge.set_value(bfi)

        # Update chart
        self.chart.append(bfi)

        # Update feature cards
        color = bfi_zone_color(bfi)
        self.card_alpha.set_value(f'{features["alpha_power"]:.2f}', color)
        self.card_theta.set_value(f'{features["theta_alpha"]:.3f}', color)
        self.card_coherence.set_value(f'{features["coherence"]:.3f}', color)
        self.card_entropy.set_value(f'{features["entropy"]:.3f}', color)
        self.card_instability.set_value(f'{features["instability"]:.4f}', color)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)

    # Global dark stylesheet
    app.setStyleSheet(f"""
        QMainWindow {{
            background-color: {COLORS['bg_dark']};
        }}
        QToolTip {{
            background-color: {COLORS['bg_card']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border']};
            font-family: Consolas;
            font-size: 10px;
        }}
    """)

    dashboard = BFIDashboard()
    dashboard.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
