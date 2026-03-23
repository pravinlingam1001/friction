import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc, FancyBboxPatch
import matplotlib.patheffects as pe
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Friction Vision",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=JetBrains+Mono:wght@400;500;600&family=Rajdhani:wght@500;600;700&family=Cinzel+Decorative:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }

.stApp {
  background-color: #07080f;
  background-image:
    radial-gradient(circle, rgba(255,140,0,0.06) 1px, transparent 1px),
    linear-gradient(rgba(255,140,0,0.015) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,140,0,0.015) 1px, transparent 1px),
    radial-gradient(ellipse 80% 50% at 50% 10%, rgba(200,80,0,0.18) 0%, rgba(120,40,0,0.08) 45%, transparent 75%),
    radial-gradient(ellipse 40% 70% at 0% 50%, rgba(100,30,0,0.12) 0%, transparent 65%),
    radial-gradient(ellipse 40% 70% at 100% 50%, rgba(100,30,0,0.10) 0%, transparent 65%);
  background-size: 28px 28px, 112px 112px, 112px 112px, 100% 100%, 100% 100%, 100% 100%;
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #060709 0%, #08090f 100%);
  border-right: 1px solid rgba(255,140,0,0.10);
  box-shadow: 4px 0 24px rgba(0,0,0,0.6);
}
section[data-testid="stSidebar"] * { color: #b08060 !important; }

.metric-card {
  background: linear-gradient(135deg, rgba(15,10,5,0.90), rgba(10,8,4,0.95));
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,140,0,0.08);
  border-top: 2px solid rgba(255,140,0,0.35);
  border-radius: 3px; padding: 20px 16px 16px;
  text-align: center; position: relative;
  transition: all 0.3s ease; box-shadow: 0 4px 16px rgba(0,0,0,0.35);
}
.metric-card:hover {
  border-top-color: rgba(255,140,0,0.65);
  box-shadow: 0 8px 28px rgba(0,0,0,0.5), 0 0 20px rgba(255,100,0,0.06);
  transform: translateY(-2px);
}
.metric-card-safe  { border-top-color: rgba(80,200,120,0.6) !important; }
.metric-card-warn  { border-top-color: rgba(255,180,0,0.6) !important; }
.metric-card-fail  { border-top-color: rgba(220,60,60,0.7) !important;
  box-shadow: 0 0 20px rgba(220,60,60,0.1) !important; }

.metric-label {
  font-size: 8px; letter-spacing: 0.22em; text-transform: uppercase;
  color: rgba(176,128,96,0.55); margin-bottom: 10px;
  font-family: 'JetBrains Mono', monospace; font-weight: 500;
}
.metric-value {
  font-size: 30px; font-weight: 600; color: #f0e0c8;
  font-family: 'Rajdhani', sans-serif; line-height: 1; letter-spacing: 0.02em;
}
.metric-unit {
  font-size: 9px; color: rgba(176,128,96,0.45);
  font-family: 'JetBrains Mono', monospace; margin-top: 6px; letter-spacing: 0.08em;
}

.check-row { display: flex; gap: 10px; margin: 16px 0 10px 0; flex-wrap: wrap; }
.check-chip {
  background: rgba(15,10,5,0.85); border: 1px solid rgba(255,140,0,0.08);
  border-radius: 3px; padding: 10px 14px; flex: 1; min-width: 160px;
}
.check-chip-label { font-size: 8px; letter-spacing: 0.22em; text-transform: uppercase;
  color: rgba(176,128,96,0.45); font-family: 'JetBrains Mono', monospace; margin-bottom: 5px; }
.check-chip-val { font-size: 20px; font-weight: 600; font-family: 'Rajdhani', sans-serif;
  color: #f0e0c8; line-height: 1; }
.check-chip-sub { font-size: 8px; color: rgba(176,128,96,0.45);
  font-family: 'JetBrains Mono', monospace; margin-top: 3px; }

.badge-move { display:inline-flex;align-items:center;gap:5px;
  background:rgba(220,60,60,0.08); color:#e07060;
  border:1px solid rgba(220,60,60,0.30); border-radius:2px;
  padding:5px 14px; font-size:9.5px; font-weight:600;
  font-family:'JetBrains Mono',monospace; letter-spacing:0.14em; text-transform:uppercase;
  animation:failPulse 2.5s ease infinite; }
.badge-static { display:inline-flex;align-items:center;gap:5px;
  background:rgba(80,200,120,0.06); color:#50c878;
  border:1px solid rgba(80,200,120,0.28); border-radius:2px;
  padding:5px 14px; font-size:9.5px; font-weight:600;
  font-family:'JetBrains Mono',monospace; letter-spacing:0.14em; text-transform:uppercase; }
.badge-impending { display:inline-flex;align-items:center;gap:5px;
  background:rgba(255,180,0,0.07); color:#c8a030;
  border:1px solid rgba(255,180,0,0.30); border-radius:2px;
  padding:5px 14px; font-size:9.5px; font-weight:600;
  font-family:'JetBrains Mono',monospace; letter-spacing:0.14em; text-transform:uppercase; }
@keyframes failPulse { 0%,100%{ box-shadow: 0 0 0 rgba(220,60,60,0); } 50%{ box-shadow: 0 0 14px rgba(220,60,60,0.2); } }

.stTabs [data-baseweb="tab-list"] { background:transparent; border-bottom:1px solid rgba(255,140,0,0.10); border-radius:0; padding:0; gap:0; }
.stTabs [data-baseweb="tab"] { border-radius:0 !important; color:rgba(176,128,96,0.45) !important;
  font-family:'JetBrains Mono',monospace !important; font-size:10px !important;
  font-weight:500 !important; padding:10px 20px !important;
  letter-spacing:0.14em !important; text-transform:uppercase !important;
  transition:all 0.2s !important; border-bottom:2px solid transparent !important; }
.stTabs [data-baseweb="tab"]:hover { color:rgba(240,224,200,0.7) !important; }
.stTabs [aria-selected="true"] { background:transparent !important; color:#f0e0c8 !important;
  border-bottom:2px solid rgba(255,140,0,0.65) !important; font-weight:600 !important; }

div[data-testid="stExpander"] {
  background:linear-gradient(135deg,rgba(10,7,3,0.92),rgba(15,10,5,0.88)) !important;
  border:1px solid rgba(255,140,0,0.06) !important;
  border-left:2px solid rgba(255,140,0,0.20) !important; border-radius:3px !important; }
div[data-testid="stExpander"]:hover { border-left-color:rgba(255,140,0,0.45) !important; }
div[data-testid="stExpander"] summary { font-family:'JetBrains Mono',monospace !important;
  font-size:10px !important; letter-spacing:0.15em !important;
  color:rgba(176,128,96,0.6) !important; text-transform:uppercase !important; }

.stSlider [data-baseweb="slider"] [role="slider"] {
  background:#ff8c00 !important; box-shadow:0 0 8px rgba(255,140,0,0.5) !important;
  border:2px solid rgba(255,140,0,0.6) !important; width:14px !important; height:14px !important; }
.stSlider [data-baseweb="slider"] [data-testid="stSliderTrackFill"] { background:rgba(255,140,0,0.45) !important; }

div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
  background:rgba(10,7,3,0.85) !important; border:1px solid rgba(255,140,0,0.10) !important;
  border-radius:3px !important; }
div[data-baseweb="select"] * { font-family:'JetBrains Mono',monospace !important;
  font-size:12px !important; color:#b08060 !important; }

.stDownloadButton > button { background:rgba(80,200,120,0.08) !important; color:#50c878 !important;
  border:1px solid rgba(80,200,120,0.25) !important; border-radius:2px !important;
  font-family:'JetBrains Mono',monospace !important; font-size:10px !important;
  font-weight:600 !important; letter-spacing:0.16em !important; text-transform:uppercase !important;
  padding:12px 28px !important; transition:all 0.25s ease !important; }
.stDownloadButton > button:hover { background:rgba(80,200,120,0.14) !important;
  border-color:rgba(80,200,120,0.45) !important; transform:translateY(-1px) !important; }

.sidebar-section { font-size:8.5px; letter-spacing:0.24em; text-transform:uppercase;
  color:rgba(255,140,0,0.50); font-family:'JetBrains Mono',monospace; font-weight:500;
  padding:4px 0 3px 10px; border-left:3px solid rgba(255,140,0,0.30); margin-bottom:10px; }

h1,h2,h3,h4 { font-family:'Cormorant Garamond',serif !important; color:#f0e0c8 !important; font-weight:400 !important; }
p, label { color:rgba(176,128,96,0.7) !important; }
hr { border-color:rgba(255,140,0,0.06) !important; }
.stSelectbox label,.stSlider label,.stNumberInput label,.stTextInput label,.stRadio label,.stCheckbox label {
  color:rgba(176,128,96,0.50) !important; font-size:9px !important; letter-spacing:0.2em !important;
  text-transform:uppercase !important; font-family:'JetBrains Mono',monospace !important; }

::-webkit-scrollbar { width:3px; height:3px; }
::-webkit-scrollbar-track { background:#07080f; }
::-webkit-scrollbar-thumb { background:rgba(255,140,0,0.20); border-radius:2px; }
</style>
""", unsafe_allow_html=True)

# ─── Color Palette ────────────────────────────────────────────────────────────
BG   = "#07080f"
SURF = "#0d0a06"
BORD = "#2a1a08"
TEXT = "#f0e0c8"
MUTE = "#6a4a28"
ACC  = "#ff8c00"
GRN  = "#50c878"
RED  = "#e07060"
YEL  = "#ffd080"
BLU  = "#60a0d0"
PURP = "#c080d0"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": SURF,
    "axes.edgecolor": BORD, "axes.labelcolor": MUTE,
    "xtick.color": MUTE, "ytick.color": MUTE,
    "grid.color": BORD, "text.color": TEXT,
    "font.family": "monospace", "font.size": 9,
})

# ─── Material Database ────────────────────────────────────────────────────────
MATERIAL_PAIRS = {
    "Steel on Steel (dry)":        {"mu_s": 0.74, "mu_k": 0.57},
    "Steel on Steel (lubricated)": {"mu_s": 0.16, "mu_k": 0.10},
    "Rubber on Concrete (dry)":    {"mu_s": 0.90, "mu_k": 0.80},
    "Wood on Wood (dry)":          {"mu_s": 0.50, "mu_k": 0.35},
    "Ice on Ice":                  {"mu_s": 0.10, "mu_k": 0.03},
    "Aluminium on Steel":          {"mu_s": 0.61, "mu_k": 0.47},
    "Copper on Steel":             {"mu_s": 0.53, "mu_k": 0.36},
    "Teflon on Steel":             {"mu_s": 0.04, "mu_k": 0.04},
    "Custom":                      {"mu_s": 0.40, "mu_k": 0.30},
}

# ─── Solvers ──────────────────────────────────────────────────────────────────

def solve_inclined_plane(W, theta_deg, mu_s, mu_k, P_applied=0.0, P_angle_deg=0.0, going_up=False):
    theta = np.radians(theta_deg)
    phi   = np.radians(P_angle_deg)
    g     = 9.81

    # Normal force (applied force may be at angle to incline surface)
    N = W * np.cos(theta) - P_applied * np.sin(phi)
    N = max(N, 0.0)

    Fs_max = mu_s * N
    Fk     = mu_k * N

    # Component of weight along incline
    W_par = W * np.sin(theta)
    P_par = P_applied * np.cos(phi)

    # Net drive force (positive = down incline)
    if going_up:
        net_drive = P_par - W_par
    else:
        net_drive = W_par - P_par

    # Determine motion status
    if going_up:
        # Force needed to push up vs max static friction
        force_needed_up = W_par + Fs_max
        motion = "SLIDES UP" if P_par > force_needed_up else \
                 ("IMPENDING (up)" if abs(P_par - force_needed_up) < 0.5 else "STATIC")
    else:
        motion = "SLIDES DOWN" if W_par > Fs_max else \
                 ("IMPENDING (down)" if abs(W_par - Fs_max) < 0.5 else "STATIC")

    phi_angle = np.degrees(np.arctan(mu_s))   # friction angle

    return {
        "N": round(N, 3),
        "Fs_max": round(Fs_max, 3),
        "Fk": round(Fk, 3),
        "W_par": round(W_par, 3),
        "P_par": round(P_par, 3),
        "motion": motion,
        "phi_angle": round(phi_angle, 2),
        "theta": theta,
        "phi": phi,
    }


def solve_belt_friction(T_tight, mu, beta_deg, direction="tightening"):
    """Capstan equation: T_tight / T_slack = e^(mu * beta)"""
    beta = np.radians(beta_deg)
    ratio = np.exp(mu * beta)
    T_slack = T_tight / ratio
    torque_transmitted = (T_tight - T_slack)  # × r for actual torque
    return {
        "T_slack": round(T_slack, 3),
        "T_tight": round(T_tight, 3),
        "ratio": round(ratio, 3),
        "beta_deg": beta_deg,
        "torque_factor": round(T_tight - T_slack, 3),
        "efficiency": round(T_slack / T_tight * 100, 1),
    }


def solve_wedge(W_load, alpha_deg, phi1_deg, phi2_deg):
    """Wedge friction — forces to raise load using a flat wedge."""
    alpha = np.radians(alpha_deg)
    phi1  = np.radians(phi1_deg)
    phi2  = np.radians(phi2_deg)
    # Force to raise load
    P_raise = W_load * np.tan(alpha + phi1 + phi2)
    # Force to lower load (self-locking if alpha < phi1+phi2)
    self_lock = alpha <= (phi1 + phi2)
    P_lower   = W_load * np.tan(phi1 + phi2 - alpha) if not self_lock else 0.0
    return {
        "P_raise": round(P_raise, 3),
        "P_lower": round(P_lower, 3),
        "self_locking": self_lock,
        "alpha_deg": alpha_deg,
    }


def solve_screw_jack(W, d_mean_mm, pitch_mm, mu_thread, collar=False, mu_collar=0.0, r_collar_mm=0.0):
    """Square-thread screw jack."""
    r = d_mean_mm / 2 / 1000  # m
    l = pitch_mm / 1000        # m
    alpha = np.arctan(l / (2 * np.pi * r))
    phi   = np.arctan(mu_thread)
    T_raise = W * r * np.tan(alpha + phi)
    T_lower = W * r * np.tan(phi - alpha) if phi > alpha else -W * r * np.tan(alpha - phi)
    self_lock = phi >= alpha

    if collar:
        T_collar = mu_collar * W * (r_collar_mm / 1000)
        T_raise  += T_collar
        T_lower  += T_collar if self_lock else -T_collar

    eff_raise = (W * l / (2 * np.pi)) / T_raise * 100 if T_raise > 0 else 0

    return {
        "T_raise": round(T_raise, 3),
        "T_lower": round(abs(T_lower), 3),
        "self_locking": self_lock,
        "lead_angle_deg": round(np.degrees(alpha), 2),
        "friction_angle_deg": round(np.degrees(phi), 2),
        "efficiency": round(eff_raise, 1),
    }


def mu_vs_angle_data(mu_s, mu_k, max_angle=85):
    angles = np.linspace(0, max_angle, 300)
    W = 100.0
    N_arr   = W * np.cos(np.radians(angles))
    Fs_arr  = mu_s * N_arr
    Fk_arr  = mu_k * N_arr
    Wpar    = W * np.sin(np.radians(angles))
    return angles, N_arr, Fs_arr, Fk_arr, Wpar


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_inclined_plane(res, theta_deg, W, P_applied, P_angle_deg, going_up):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURF)

    theta = res["theta"]
    # Draw inclined plane
    plane_len = 5.0
    px = [0, plane_len * np.cos(theta)]
    py = [0, plane_len * np.sin(theta)]
    ax.fill([0, plane_len*np.cos(theta), plane_len*np.cos(theta), 0],
            [0, plane_len*np.sin(theta), 0, 0], color=BORD, alpha=0.7, zorder=1)
    ax.plot([0, plane_len*np.cos(theta)], [0, plane_len*np.sin(theta)],
            color=ACC, lw=2.0, zorder=2)
    ax.plot([0, plane_len*np.cos(theta), 0, 0],
            [0, plane_len*np.sin(theta), plane_len*np.sin(theta)-0, 0],
            color=BORD, lw=0.8, alpha=0.4)

    # Block on incline
    bx = plane_len * 0.5 * np.cos(theta)
    by = plane_len * 0.5 * np.sin(theta)
    block_w, block_h = 0.55, 0.40
    corners = np.array([[-block_w/2, 0], [block_w/2, 0],
                         [block_w/2, block_h], [-block_w/2, block_h]])
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    corners_rot = corners @ rot.T
    corners_rot[:, 0] += bx
    corners_rot[:, 1] += by
    poly = plt.Polygon(corners_rot, closed=True, facecolor="#1a1208",
                       edgecolor=ACC, linewidth=1.5, zorder=5)
    ax.add_patch(poly)

    cx = bx + (block_h/2)*(-np.sin(theta)) + 0
    cy = by + (block_h/2)*( np.cos(theta))
    cx_top = bx + block_h*(-np.sin(theta))
    cy_top = by + block_h*( np.cos(theta))

    scale = 0.45
    # Weight arrow (straight down)
    ax.annotate("", xy=(cx, cy - W*scale/200), xytext=(cx, cy + W*scale/200 * 0.3),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.0, mutation_scale=14))
    ax.text(cx + 0.08, cy - W*scale/200 - 0.08, f"W={W:.0f} N",
            fontsize=8.5, color=RED, fontweight="bold")

    # Normal force
    nx = -np.sin(theta) * res["N"] / 200 * scale
    ny =  np.cos(theta) * res["N"] / 200 * scale
    ax.annotate("", xy=(cx_top + nx*1.6, cy_top + ny*1.6),
                xytext=(cx_top, cy_top),
                arrowprops=dict(arrowstyle="-|>", color=BLU, lw=1.8, mutation_scale=12))
    ax.text(cx_top + nx*1.8, cy_top + ny*1.8, f"N={res['N']:.1f} N",
            fontsize=8, color=BLU)

    # Friction force
    dir_sign = 1.0 if going_up else -1.0
    motion_active = "SLIDES" in res["motion"]
    Ff_val = res["Fk"] if motion_active else res["Fs_max"]
    Ff_col = YEL if not motion_active else RED
    fx_f = np.cos(theta) * Ff_val / 200 * scale * dir_sign
    fy_f = np.sin(theta) * Ff_val / 200 * scale * dir_sign
    ax.annotate("", xy=(cx - fx_f*1.8, cy - fy_f*1.8), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color=Ff_col, lw=1.8, mutation_scale=12))
    ax.text(cx - fx_f*2.2, cy - fy_f*2.2,
            f"{'Fk' if motion_active else 'Fs'}={Ff_val:.1f} N",
            fontsize=8, color=Ff_col)

    # Applied force P
    if P_applied > 0:
        phi = res["phi"]
        inc_up = np.array([np.cos(theta), np.sin(theta)])
        perp   = np.array([-np.sin(theta), np.cos(theta)])
        P_vec  = P_applied * (np.cos(phi)*inc_up + np.sin(phi)*perp)
        P_sc   = P_vec / 200 * scale * (1.0 if going_up else -1.0)
        ax.annotate("", xy=(cx + P_sc[0]*2.0, cy + P_sc[1]*2.0),
                    xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="-|>", color=GRN, lw=1.8, mutation_scale=12))
        ax.text(cx + P_sc[0]*2.3, cy + P_sc[1]*2.3, f"P={P_applied:.0f} N",
                fontsize=8, color=GRN)

    # Angle arc
    arc = Arc((0, 0), 1.4, 1.4, angle=0, theta1=0, theta2=theta_deg,
              color=YEL, lw=1.0, linestyle="--")
    ax.add_patch(arc)
    ax.text(0.85, 0.12, f"θ={theta_deg}°", fontsize=8.5, color=YEL)

    # Motion badge
    badge_col = RED if "SLIDES" in res["motion"] else (YEL if "IMPENDING" in res["motion"] else GRN)
    motion_icon = "▶" if "SLIDES" in res["motion"] else ("⚡" if "IMPENDING" in res["motion"] else "■")
    ax.text(0.98, 0.97, f"{motion_icon} {res['motion']}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color=badge_col, fontweight="bold",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BG, edgecolor=badge_col, alpha=0.85))

    # Friction angle arc on block
    ax.text(2.8, 0.05, f"φ (friction angle) = {res['phi_angle']}°",
            fontsize=8, color=MUTE, fontstyle="italic")

    ax.set_xlim(-0.4, 6.2)
    ax.set_ylim(-0.5, plane_len * np.sin(theta) + 0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=9, color=MUTE)
    ax.set_ylabel("y (m)", fontsize=9, color=MUTE)
    ax.set_title("INCLINED PLANE — Free Body Diagram", fontsize=10, color=TEXT,
                 fontweight="bold", loc="left", pad=8)
    ax.grid(alpha=0.10, lw=0.4, linestyle=":")
    for sp in ax.spines.values(): sp.set_edgecolor(BORD)
    plt.tight_layout(pad=0.5)
    return fig


def plot_force_vs_angle(angles, N_arr, Fs_arr, Fk_arr, Wpar, mu_s, mu_k):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURF)

    ax.plot(angles, Wpar,  color=RED, lw=1.8, label="W·sin(θ)  [drive]", zorder=3)
    ax.plot(angles, Fs_arr, color=YEL, lw=1.5, linestyle="--", label=f"μs·N  [static limit]  μs={mu_s}", zorder=3)
    ax.plot(angles, Fk_arr, color=ACC, lw=1.2, linestyle=":", label=f"μk·N  [kinetic]  μk={mu_k}", zorder=3)
    ax.plot(angles, N_arr,  color=BLU, lw=1.2, alpha=0.7, label="Normal N = W·cos(θ)", zorder=2)

    # Critical angle where sliding starts
    crit_idx = np.argmin(np.abs(Wpar - Fs_arr))
    crit_ang = angles[crit_idx]
    ax.axvline(crit_ang, color=GRN, lw=1.0, linestyle="--", alpha=0.7)
    ax.scatter([crit_ang], [Wpar[crit_idx]], color=GRN, s=60, zorder=6)
    ax.text(crit_ang + 0.8, Wpar[crit_idx] + 1.5,
            f"θ_crit = {crit_ang:.1f}°\n(≈ arctan μs)",
            fontsize=7.5, color=GRN, fontweight="bold")

    ax.fill_between(angles, Wpar, Fs_arr, where=(Wpar > Fs_arr),
                    color=RED, alpha=0.12, label="Sliding zone")
    ax.fill_between(angles, Wpar, Fs_arr, where=(Wpar <= Fs_arr),
                    color=GRN, alpha=0.08, label="Static zone")

    ax.set_xlim(0, max(angles))
    ax.set_ylim(-2, 105)
    ax.set_xlabel("Inclination angle θ (degrees)", fontsize=9, color=MUTE)
    ax.set_ylabel("Force (N)  [per 100 N weight]", fontsize=9, color=MUTE)
    ax.set_title("FORCE vs INCLINATION ANGLE  |  W = 100 N reference",
                 fontsize=10, color=TEXT, fontweight="bold", loc="left", pad=8)
    ax.legend(fontsize=7.5, facecolor=SURF, edgecolor=BORD, labelcolor=TEXT,
              loc="upper left", framealpha=0.85)
    ax.grid(alpha=0.12, lw=0.4, linestyle=":")
    for sp in ax.spines.values(): sp.set_edgecolor(BORD)
    plt.tight_layout(pad=0.5)
    return fig


def plot_belt_friction(res, mu, beta_deg):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [1.3, 1]})
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURF)
        for sp in ax.spines.values(): sp.set_edgecolor(BORD)

    # ── Belt wrap diagram ──────────────────────────────────────────────────
    R = 1.0
    beta = np.radians(beta_deg)
    start_ang = np.pi / 2 - beta / 2
    end_ang   = np.pi / 2 + beta / 2

    # Pulley
    theta_full = np.linspace(0, 2*np.pi, 200)
    ax1.fill(R * np.cos(theta_full), R * np.sin(theta_full), color="#1a1208",
             alpha=0.9, zorder=1)
    ax1.plot(R * np.cos(theta_full), R * np.sin(theta_full), color=ACC, lw=1.5, zorder=2)

    # Belt contact arc (highlight)
    theta_arc = np.linspace(start_ang, end_ang, 100)
    belt_r = R * 1.08
    ax1.fill_between(belt_r*np.cos(theta_arc), belt_r*np.sin(theta_arc),
                     R*np.cos(theta_arc), R*np.sin(theta_arc),
                     color=YEL, alpha=0.35)
    ax1.plot(belt_r*np.cos(theta_arc), belt_r*np.sin(theta_arc),
             color=YEL, lw=2.0, zorder=3, label=f"Contact arc β={beta_deg}°")

    # T_tight arrow
    tx_t = R * 1.2 * np.cos(start_ang)
    ty_t = R * 1.2 * np.sin(start_ang)
    dx_t = -0.6 * np.sin(start_ang)
    dy_t =  0.6 * np.cos(start_ang)
    ax1.annotate("", xy=(tx_t + dx_t, ty_t + dy_t), xytext=(tx_t, ty_t),
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=2, mutation_scale=14))
    ax1.text(tx_t + dx_t * 1.2, ty_t + dy_t * 1.2,
             f"T_tight\n{res['T_tight']:.1f} N", fontsize=8, color=RED, ha="center")

    # T_slack arrow
    tx_s = R * 1.2 * np.cos(end_ang)
    ty_s = R * 1.2 * np.sin(end_ang)
    dx_s =  0.6 * np.sin(end_ang)
    dy_s = -0.6 * np.cos(end_ang)
    ax1.annotate("", xy=(tx_s + dx_s, ty_s + dy_s), xytext=(tx_s, ty_s),
                 arrowprops=dict(arrowstyle="-|>", color=GRN, lw=2, mutation_scale=14))
    ax1.text(tx_s + dx_s * 1.2, ty_s + dy_s * 1.2,
             f"T_slack\n{res['T_slack']:.1f} N", fontsize=8, color=GRN, ha="center")

    # Angle arc
    mid_ang = (start_ang + end_ang) / 2
    ax1.annotate("", xy=(0.6*np.cos(start_ang), 0.6*np.sin(start_ang)),
                 xytext=(0.6*np.cos(end_ang), 0.6*np.sin(end_ang)),
                 arrowprops=dict(arrowstyle="<->", color=YEL, lw=0.8))
    ax1.text(0.0, 0.0, f"β={beta_deg}°", ha="center", va="center",
             fontsize=9, color=YEL, fontweight="bold")

    ax1.set_xlim(-2.1, 2.1)
    ax1.set_ylim(-2.1, 2.1)
    ax1.set_aspect("equal")
    ax1.set_title("CAPSTAN EQUATION  |  Belt-Pulley FBD", fontsize=9,
                  color=TEXT, fontweight="bold", loc="left", pad=6)
    ax1.legend(fontsize=7.5, facecolor=SURF, edgecolor=BORD, labelcolor=TEXT)
    ax1.grid(alpha=0.08, lw=0.3)

    # ── T ratio vs beta ────────────────────────────────────────────────────
    betas = np.linspace(0, 360, 300)
    ratios = np.exp(mu * np.radians(betas))
    ax2.plot(betas, ratios, color=ACC, lw=1.8, zorder=3)
    ax2.scatter([beta_deg], [res["ratio"]], color=YEL, s=70, zorder=5)
    ax2.axvline(beta_deg, color=YEL, lw=0.8, linestyle="--", alpha=0.6)
    ax2.text(beta_deg + 5, res["ratio"] * 1.04,
             f"β={beta_deg}°\nT₁/T₂={res['ratio']:.2f}", fontsize=7.5, color=YEL)
    ax2.fill_between(betas, 1, ratios, color=ACC, alpha=0.12)
    ax2.set_xlabel("Wrap angle β (degrees)", fontsize=9, color=MUTE)
    ax2.set_ylabel("Tension ratio T_tight / T_slack", fontsize=9, color=MUTE)
    ax2.set_title(f"TENSION RATIO  |  μ = {mu}", fontsize=9,
                  color=TEXT, fontweight="bold", loc="left", pad=6)
    ax2.grid(alpha=0.12, lw=0.4, linestyle=":")
    plt.tight_layout(pad=0.5)
    return fig


def plot_screw_jack(res, W, d_mean_mm, pitch_mm):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [1, 1.2]})
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURF)
        for sp in ax.spines.values(): sp.set_edgecolor(BORD)

    # ── Screw geometry schematic ──────────────────────────────────────────
    n_threads = 7
    r = d_mean_mm / 2
    screw_h = pitch_mm * n_threads

    # Body
    ax1.fill([-r, r, r, -r], [0, 0, screw_h, screw_h],
             facecolor="#1a1208", edgecolor=ACC, lw=1.5, alpha=0.9)

    # Thread spirals (simplified)
    for i in range(n_threads + 1):
        y = i * pitch_mm
        ax1.plot([-r*1.18, -r], [y, y], color=MUTE, lw=1.0)
        ax1.plot([r, r*1.18], [y, y], color=MUTE, lw=1.0)

    # Helix angle arc at bottom thread
    arc_r = r * 0.7
    helix_ang = res["lead_angle_deg"]
    ax1.annotate("", xy=(arc_r * np.cos(np.radians(90 - helix_ang)),
                          arc_r * np.sin(np.radians(90 - helix_ang))),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=YEL, lw=1.2))
    ax1.text(arc_r * 0.6, -pitch_mm * 0.6,
             f"α={helix_ang}°\n(lead angle)", fontsize=7.5, color=YEL, ha="center")

    # Load arrow
    ax1.annotate("", xy=(0, screw_h + pitch_mm * 0.8),
                 xytext=(0, screw_h + pitch_mm * 2.2),
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=2, mutation_scale=14))
    ax1.text(r * 1.4, screw_h + pitch_mm * 1.5, f"W={W:.0f} N",
             fontsize=8.5, color=RED, fontweight="bold")

    # Torque arrow (curved)
    t_arc = Arc((0, pitch_mm), r * 0.9, r * 0.9, angle=0,
                theta1=0, theta2=270, color=ACC, lw=1.5)
    ax1.add_patch(t_arc)
    ax1.text(r * 1.5, pitch_mm * 0.5, f"T_raise\n{res['T_raise']:.2f} N·m",
             fontsize=8, color=ACC)

    ax1.set_xlim(-r * 2.5, r * 2.8)
    ax1.set_ylim(-pitch_mm * 1.5, screw_h + pitch_mm * 2.8)
    ax1.set_aspect("equal")
    ax1.set_xlabel("r (mm)", fontsize=9, color=MUTE)
    ax1.set_ylabel("height (mm)", fontsize=9, color=MUTE)
    ax1.set_title("SCREW JACK SCHEMATIC", fontsize=9,
                  color=TEXT, fontweight="bold", loc="left", pad=6)
    ax1.grid(alpha=0.08, lw=0.3)

    # ── Torque vs Lead angle ───────────────────────────────────────────────
    alpha_range = np.linspace(0.5, 45, 200)
    phi_rad     = np.radians(res["friction_angle_deg"])
    r_m         = d_mean_mm / 2 / 1000
    T_raise_arr = W * r_m * np.tan(np.radians(alpha_range) + phi_rad)
    T_lower_arr = W * r_m * np.abs(np.tan(phi_rad - np.radians(alpha_range)))
    eff_arr     = np.tan(np.radians(alpha_range)) / np.tan(np.radians(alpha_range) + phi_rad) * 100

    ax2.plot(alpha_range, T_raise_arr, color=RED, lw=1.8, label="Torque (raise)")
    ax2.plot(alpha_range, T_lower_arr, color=GRN, lw=1.5, linestyle="--", label="Torque (lower)")
    ax2_r = ax2.twinx()
    ax2_r.set_facecolor(SURF)
    ax2_r.plot(alpha_range, eff_arr, color=YEL, lw=1.2, linestyle=":", label="Efficiency %")
    ax2_r.set_ylabel("Efficiency (%)", fontsize=8, color=YEL)
    ax2_r.yaxis.label.set_color(YEL)
    ax2_r.tick_params(colors=YEL)

    ax2.axvline(res["lead_angle_deg"], color=ACC, lw=1.0, linestyle="--", alpha=0.8)
    ax2.scatter([res["lead_angle_deg"]], [res["T_raise"]], color=ACC, s=60, zorder=5)
    ax2.text(res["lead_angle_deg"] + 0.5, res["T_raise"] * 1.05,
             f"α={res['lead_angle_deg']}°\nη={res['efficiency']}%", fontsize=7.5, color=ACC)

    ax2.set_xlabel("Lead angle α (degrees)", fontsize=9, color=MUTE)
    ax2.set_ylabel("Torque (N·m)", fontsize=9, color=MUTE)
    ax2.set_title("TORQUE & EFFICIENCY vs LEAD ANGLE", fontsize=9,
                  color=TEXT, fontweight="bold", loc="left", pad=6)
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2,
               fontsize=7.5, facecolor=SURF, edgecolor=BORD, labelcolor=TEXT)
    ax2.grid(alpha=0.10, lw=0.4, linestyle=":")
    plt.tight_layout(pad=0.5)
    return fig


def plot_friction_circle(mu_s, mu_k, N_list):
    """Visualise friction cone and friction circle for multiple normal loads."""
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURF)
    for sp in ax.spines.values(): sp.set_edgecolor(BORD)

    phi_s = np.arctan(mu_s)
    phi_k = np.arctan(mu_k)

    colors = [ACC, GRN, BLU, PURP]
    for i, N in enumerate(N_list[:4]):
        r_s = mu_s * N
        r_k = mu_k * N
        col = colors[i % len(colors)]
        c_s = plt.Circle((0, N), r_s, fill=False, color=col, lw=1.4,
                          linestyle="--", label=f"Static (N={N}N, Fs={r_s:.1f}N)")
        c_k = plt.Circle((0, N), r_k, fill=False, color=col, lw=0.8,
                          linestyle=":", alpha=0.6)
        ax.add_patch(c_s)
        ax.add_patch(c_k)
        ax.scatter([0], [N], color=col, s=20, zorder=4)

    # Friction cone lines
    max_N   = max(N_list) if N_list else 100
    cone_ys = np.array([0, max_N * 1.2])
    ax.plot(np.tan(phi_s) * cone_ys,  cone_ys, color=YEL, lw=1.2, linestyle="--",
            label=f"Friction cone (φs={np.degrees(phi_s):.1f}°)")
    ax.plot(-np.tan(phi_s) * cone_ys, cone_ys, color=YEL, lw=1.2, linestyle="--")
    ax.plot(np.tan(phi_k) * cone_ys,  cone_ys, color=MUTE, lw=0.8, linestyle=":")
    ax.plot(-np.tan(phi_k) * cone_ys, cone_ys, color=MUTE, lw=0.8, linestyle=":")

    ax.axhline(0, color=BORD, lw=0.8)
    ax.axvline(0, color=BORD, lw=0.8, linestyle="--", alpha=0.5)

    span = max_N * 1.4
    ax.set_xlim(-span * 0.6, span * 0.6)
    ax.set_ylim(-span * 0.15, span * 1.3)
    ax.set_xlabel("Friction Force F (N)", fontsize=9, color=MUTE)
    ax.set_ylabel("Normal Force N (N)", fontsize=9, color=MUTE)
    ax.set_title("FRICTION CIRCLE & FRICTION CONE", fontsize=10,
                 color=TEXT, fontweight="bold", loc="left", pad=8)
    ax.legend(fontsize=7.5, facecolor=SURF, edgecolor=BORD, labelcolor=TEXT)
    ax.grid(alpha=0.12, lw=0.4, linestyle=":")
    ax.text(0.02, 0.02, f"μs={mu_s} | μk={mu_k} | φs={np.degrees(phi_s):.1f}°",
            transform=ax.transAxes, fontsize=8, color=MUTE, fontfamily="monospace")
    plt.tight_layout(pad=0.5)
    return fig


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 0 4px 0;'>
      <div style='font-size:18px;font-weight:700;color:#f0e0c8;
        font-family:Rajdhani,sans-serif;letter-spacing:0.05em;'>⚙️ Friction Setup</div>
      <div style='height:1px;background:linear-gradient(90deg,rgba(255,140,0,0.6),transparent);
        margin-top:8px;'></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'>🧱 Material Pair</div>", unsafe_allow_html=True)
    mat_pair = st.selectbox("Material Combination", list(MATERIAL_PAIRS.keys()))
    mu_data  = MATERIAL_PAIRS[mat_pair]

    if mat_pair == "Custom":
        mu_s = st.slider("μs — Static Coeff.", 0.01, 1.50, 0.40, 0.01)
        mu_k = st.slider("μk — Kinetic Coeff.", 0.01, 1.50, 0.30, 0.01)
    else:
        mu_s = mu_data["mu_s"]
        mu_k = mu_data["mu_k"]
        st.markdown(f"""
        <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#b08060;
                    padding:8px;background:rgba(15,10,5,0.8);border-radius:3px;
                    border-left:2px solid rgba(255,140,0,0.3);'>
          μs = {mu_s} &nbsp;|&nbsp; μk = {mu_k}<br>
          φ (friction angle) = {np.degrees(np.arctan(mu_s)):.1f}°
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        "<div style='height:1px;background:linear-gradient(90deg,rgba(255,140,0,0.15),transparent);"
        "margin:12px 0 10px 0;'></div><div class='sidebar-section'>🏔️ Inclined Plane</div>",
        unsafe_allow_html=True
    )
    W         = st.slider("Weight W (N)", 10.0, 2000.0, 100.0, 10.0)
    theta_deg = st.slider("Inclination θ (°)", 0.0, 85.0, 30.0, 1.0)
    going_up  = st.checkbox("Applied force pushes UP the slope", value=False)
    P_applied = st.slider("Applied Force P (N)", 0.0, 500.0, 0.0, 5.0)
    P_angle   = st.slider("P angle to incline surface (°)", 0.0, 45.0, 0.0, 1.0)

    st.markdown(
        "<div style='height:1px;background:linear-gradient(90deg,rgba(255,140,0,0.15),transparent);"
        "margin:12px 0 10px 0;'></div><div class='sidebar-section'>⚙️ Belt Friction</div>",
        unsafe_allow_html=True
    )
    T_tight   = st.slider("Tight-side tension T₁ (N)", 10.0, 5000.0, 500.0, 10.0)
    beta_deg  = st.slider("Wrap angle β (°)", 10.0, 360.0, 180.0, 5.0)

    st.markdown(
        "<div style='height:1px;background:linear-gradient(90deg,rgba(255,140,0,0.15),transparent);"
        "margin:12px 0 10px 0;'></div><div class='sidebar-section'>🔩 Screw Jack</div>",
        unsafe_allow_html=True
    )
    W_jack     = st.slider("Load W (N)", 100.0, 50000.0, 5000.0, 100.0)
    d_mean_mm  = st.slider("Mean thread diameter (mm)", 10.0, 100.0, 40.0, 1.0)
    pitch_mm   = st.slider("Pitch (mm)", 1.0, 20.0, 6.0, 0.5)
    mu_thread  = st.slider("Thread friction μ_thread", 0.01, 0.60, 0.18, 0.01)

    st.markdown(
        "<div style='height:1px;background:linear-gradient(90deg,rgba(255,140,0,0.15),transparent);"
        "margin:12px 0 10px 0;'></div><div class='sidebar-section'>⭕ Friction Circle</div>",
        unsafe_allow_html=True
    )
    n_circles = st.number_input("Number of Normal loads to plot", 1, 4, 3)
    circle_loads = []
    for i in range(int(n_circles)):
        circle_loads.append(st.number_input(f"N{i+1} (N)", 10.0, 5000.0,
                                            float(100 * (i + 1)), 10.0, key=f"cn{i}"))

    st.info("⚡ All results update live.")


# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:36px 0 20px 0; text-align:center;'>
  <div style='font-size:11px; letter-spacing:0.36em; text-transform:uppercase;
    color:rgba(255,140,0,0.50); font-family:JetBrains Mono,monospace;
    font-weight:500; margin-bottom:16px;'>Engineering Mechanics Suite</div>
  <div style='font-size:68px; font-weight:300; font-family:Cormorant Garamond,serif;
    color:#f0e0c8; line-height:1; letter-spacing:-0.02em; margin-bottom:6px;'>
    Friction Vision</div>
  <div style='font-size:22px; font-weight:300; font-style:italic;
    font-family:Cormorant Garamond,serif; color:rgba(255,140,0,0.55);
    letter-spacing:0.04em; margin-bottom:24px;'>Friction &amp; Impending Motion Analyzer</div>
  <div style='display:flex; align-items:center; justify-content:center;
    gap:14px; margin-bottom:12px;'>
    <div style='height:1px; width:100px; background:rgba(255,140,0,0.20);'></div>
    <div style='width:5px; height:5px; border-radius:50%; background:rgba(255,140,0,0.55);'></div>
    <div style='height:1px; width:100px; background:rgba(255,140,0,0.20);'></div>
  </div>
</div>
<div style='height:1px; background:linear-gradient(90deg,transparent,rgba(255,140,0,0.28),transparent);
  margin-bottom:28px;'></div>
""", unsafe_allow_html=True)


# ─── Solve all ────────────────────────────────────────────────────────────────
try:
    res_inc  = solve_inclined_plane(W, theta_deg, mu_s, mu_k, P_applied, P_angle, going_up)
    res_belt = solve_belt_friction(T_tight, mu_s, beta_deg)
    res_jack = solve_screw_jack(W_jack, d_mean_mm, pitch_mm, mu_thread)
    angles, N_arr, Fs_arr, Fk_arr, Wpar = mu_vs_angle_data(mu_s, mu_k)

    # ── Key metric cards ──────────────────────────────────────────────────
    cols = st.columns(5)
    cards = [
        ("Normal Force N",    f"{res_inc['N']:.1f}",     "N"),
        ("Max Static Fs",     f"{res_inc['Fs_max']:.1f}", "N"),
        ("Kinetic Friction",  f"{res_inc['Fk']:.1f}",    "N"),
        ("Friction Angle φ",  f"{res_inc['phi_angle']}", "degrees"),
        ("Belt Ratio T₁/T₂",  f"{res_belt['ratio']:.3f}", "e^(μβ)"),
    ]
    sf_classes = ["", "", "", "", ""]
    for i, (col, (label, val, unit)) in enumerate(zip(cols, cards)):
        with col:
            st.markdown(f"""<div class='metric-card {sf_classes[i]}'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{val}</div>
                <div class='metric-unit'>{unit}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── IS code / motion status bar ───────────────────────────────────────
    motion = res_inc["motion"]
    if "SLIDES" in motion:
        mbadge = f"<span class='badge-move'>▶ MOTION — {motion}</span>"
        mc     = RED
    elif "IMPENDING" in motion:
        mbadge = f"<span class='badge-impending'>⚡ IMPENDING MOTION — {motion}</span>"
        mc     = YEL
    else:
        mbadge = f"<span class='badge-static'>■ STATIC EQUILIBRIUM</span>"
        mc     = GRN

    sl_str   = "✓ SELF-LOCKING" if res_jack["self_locking"] else "✗ NOT SELF-LOCKING"
    sl_color = GRN if res_jack["self_locking"] else RED

    st.markdown(f"""
    <div class='check-row'>
      <div class='check-chip'>
        <div class='check-chip-label'>Inclined Plane Status</div>
        <div style='margin-top:4px'>{mbadge}</div>
        <div class='check-chip-sub'>W∥={res_inc['W_par']:.1f} N  |  Fs_max={res_inc['Fs_max']:.1f} N</div>
      </div>
      <div class='check-chip'>
        <div class='check-chip-label'>Screw Jack · Torque to Raise</div>
        <div class='check-chip-val' style='color:{ACC};'>{res_jack['T_raise']:.2f}</div>
        <div class='check-chip-sub'>N·m  |  η={res_jack['efficiency']}%  |  α={res_jack['lead_angle_deg']}°</div>
      </div>
      <div class='check-chip'>
        <div class='check-chip-label'>Screw Jack · Self-Locking</div>
        <div class='check-chip-val' style='color:{sl_color};font-size:14px;'>{sl_str}</div>
        <div class='check-chip-sub'>φ={res_jack['friction_angle_deg']}°  α={res_jack['lead_angle_deg']}°</div>
      </div>
      <div class='check-chip'>
        <div class='check-chip-label'>Belt · Slack-Side Tension</div>
        <div class='check-chip-val' style='color:{GRN};'>{res_belt['T_slack']:.1f}</div>
        <div class='check-chip-sub'>N  |  Force transmitted: {res_belt['torque_factor']:.1f} N</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏔️  Inclined Plane",
        "📈  Force vs Angle",
        "🔄  Belt Friction",
        "🔩  Screw Jack",
        "⭕  Friction Circle",
        "📋  Summary",
    ])

    with tab1:
        st.markdown("""<div style='font-family:JetBrains Mono,monospace;font-size:9px;
            letter-spacing:0.16em;text-transform:uppercase;color:rgba(255,140,0,0.40);
            margin-bottom:10px;'>Free body diagram — inclined plane with all contact forces</div>""",
            unsafe_allow_html=True)
        fig1 = plot_inclined_plane(res_inc, theta_deg, W, P_applied, P_angle, going_up)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

        c1, c2, c3, c4 = st.columns(4)
        eqs = [
            ("N = W·cos θ − P·sin φ", f"{res_inc['N']:.2f} N"),
            ("Fs_max = μs·N",          f"{res_inc['Fs_max']:.2f} N"),
            ("Fk = μk·N",              f"{res_inc['Fk']:.2f} N"),
            ("φ = arctan(μs)",         f"{res_inc['phi_angle']}°"),
        ]
        for col, (eq, val) in zip([c1,c2,c3,c4], eqs):
            with col:
                st.markdown(f"""
                <div style='background:rgba(10,7,3,0.85);border:1px solid rgba(255,140,0,0.10);
                  border-radius:3px;padding:10px 12px;text-align:center;'>
                  <div style='font-family:JetBrains Mono,monospace;font-size:8px;
                    color:rgba(176,128,96,0.5);letter-spacing:0.14em;margin-bottom:6px;'>{eq}</div>
                  <div style='font-family:Rajdhani,sans-serif;font-size:22px;
                    font-weight:600;color:#f0e0c8;'>{val}</div>
                </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("""<div style='font-family:JetBrains Mono,monospace;font-size:9px;
            letter-spacing:0.16em;text-transform:uppercase;color:rgba(255,140,0,0.40);
            margin-bottom:10px;'>How friction and normal forces vary with slope angle (W=100N reference)</div>""",
            unsafe_allow_html=True)
        crit_angle = np.degrees(np.arctan(mu_s))
        st.markdown(f"""
        <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#b08060;
                    padding:8px 12px;background:rgba(10,7,3,0.8);border-radius:3px;
                    border-left:2px solid rgba(255,140,0,0.4);margin-bottom:12px;display:inline-block;'>
          Critical angle θ_crit = arctan(μs) = arctan({mu_s}) = <span style='color:#f0e0c8;font-weight:600;'>{crit_angle:.1f}°</span>
          &nbsp;|&nbsp; Above this angle: block slides without any applied force
        </div>""", unsafe_allow_html=True)
        fig2 = plot_force_vs_angle(angles, N_arr, Fs_arr, Fk_arr, Wpar, mu_s, mu_k)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    with tab3:
        st.markdown("""<div style='font-family:JetBrains Mono,monospace;font-size:9px;
            letter-spacing:0.16em;text-transform:uppercase;color:rgba(255,140,0,0.40);
            margin-bottom:10px;'>Capstan equation: T₁/T₂ = e^(μβ)</div>""",
            unsafe_allow_html=True)
        fig3 = plot_belt_friction(res_belt, mu_s, beta_deg)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        b1, b2, b3 = st.columns(3)
        belt_vals = [
            ("T_tight (T₁)", f"{res_belt['T_tight']:.1f} N",   RED),
            ("T_slack (T₂)", f"{res_belt['T_slack']:.1f} N",   GRN),
            ("Force diff",   f"{res_belt['torque_factor']:.1f} N", YEL),
        ]
        for col, (label, val, col_c) in zip([b1,b2,b3], belt_vals):
            with col:
                st.markdown(f"""
                <div style='background:rgba(10,7,3,0.85);border:1px solid rgba(255,140,0,0.10);
                  border-radius:3px;padding:10px 12px;text-align:center;'>
                  <div style='font-family:JetBrains Mono,monospace;font-size:8px;
                    color:rgba(176,128,96,0.5);letter-spacing:0.14em;margin-bottom:6px;'>{label}</div>
                  <div style='font-family:Rajdhani,sans-serif;font-size:22px;
                    font-weight:600;color:{col_c};'>{val}</div>
                </div>""", unsafe_allow_html=True)

    with tab4:
        st.markdown("""<div style='font-family:JetBrains Mono,monospace;font-size:9px;
            letter-spacing:0.16em;text-transform:uppercase;color:rgba(255,140,0,0.40);
            margin-bottom:10px;'>Square-thread screw jack analysis — torque, efficiency & self-locking</div>""",
            unsafe_allow_html=True)
        fig4 = plot_screw_jack(res_jack, W_jack, d_mean_mm, pitch_mm)
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

        j1, j2, j3, j4 = st.columns(4)
        jack_vals = [
            ("Torque (Raise)", f"{res_jack['T_raise']:.2f} N·m", ACC),
            ("Torque (Lower)", f"{res_jack['T_lower']:.2f} N·m", GRN),
            ("Efficiency",     f"{res_jack['efficiency']}%",      YEL),
            ("Self-locking",   "YES" if res_jack["self_locking"] else "NO",
             GRN if res_jack["self_locking"] else RED),
        ]
        for col, (label, val, col_c) in zip([j1,j2,j3,j4], jack_vals):
            with col:
                st.markdown(f"""
                <div style='background:rgba(10,7,3,0.85);border:1px solid rgba(255,140,0,0.10);
                  border-radius:3px;padding:10px 12px;text-align:center;'>
                  <div style='font-family:JetBrains Mono,monospace;font-size:8px;
                    color:rgba(176,128,96,0.5);letter-spacing:0.14em;margin-bottom:6px;'>{label}</div>
                  <div style='font-family:Rajdhani,sans-serif;font-size:22px;
                    font-weight:600;color:{col_c};'>{val}</div>
                </div>""", unsafe_allow_html=True)

        with st.expander("📐 Screw Jack Theory"):
            st.markdown(f"""
            <div style='font-family:JetBrains Mono,monospace;font-size:10px;
                        color:#b08060;line-height:1.9;'>
              Lead angle &nbsp;&nbsp;&nbsp;α = arctan(l / 2πr) = <b style='color:#f0e0c8'>{res_jack['lead_angle_deg']}°</b><br>
              Friction angle &nbsp;φ = arctan(μ) &nbsp;&nbsp;&nbsp;&nbsp;= <b style='color:#f0e0c8'>{res_jack['friction_angle_deg']}°</b><br>
              T_raise = W·r·tan(α + φ) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <b style='color:{ACC}'>{res_jack['T_raise']:.3f} N·m</b><br>
              T_lower = W·r·tan(φ − α) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <b style='color:{GRN}'>{res_jack['T_lower']:.3f} N·m</b><br>
              Self-locking condition: φ ≥ α → {'<b style="color:'+GRN+'">YES ✓</b>' if res_jack['self_locking'] else '<b style="color:'+RED+'">NO ✗</b>'}
            </div>""", unsafe_allow_html=True)

    with tab5:
        st.markdown("""<div style='font-family:JetBrains Mono,monospace;font-size:9px;
            letter-spacing:0.16em;text-transform:uppercase;color:rgba(255,140,0,0.40);
            margin-bottom:10px;'>Friction circle radius = μ·N  |  Resultant reaction must lie within cone</div>""",
            unsafe_allow_html=True)
        fig5 = plot_friction_circle(mu_s, mu_k, circle_loads)
        st.pyplot(fig5, use_container_width=True)
        plt.close(fig5)

    with tab6:
        st.markdown("""<div style='font-family:JetBrains Mono,monospace;font-size:9px;
            letter-spacing:0.16em;text-transform:uppercase;color:rgba(255,140,0,0.40);
            margin-bottom:10px;'>Complete engineering summary — all modules</div>""",
            unsafe_allow_html=True)

        summary_data = {
            "Module": [
                "Inclined Plane", "Inclined Plane", "Inclined Plane",
                "Inclined Plane", "Inclined Plane", "Inclined Plane",
                "Belt Friction", "Belt Friction", "Belt Friction",
                "Screw Jack", "Screw Jack", "Screw Jack", "Screw Jack",
                "Material Pair",
            ],
            "Parameter": [
                "Weight W", "Inclination θ", "Normal Force N",
                "Max Static Friction Fs", "Kinetic Friction Fk", "Motion Status",
                "Tight Tension T₁", "Slack Tension T₂", "Tension Ratio T₁/T₂",
                "Torque to Raise", "Torque to Lower", "Efficiency", "Self-Locking",
                "Material Pair",
            ],
            "Value": [
                f"{W} N", f"{theta_deg}°", f"{res_inc['N']:.2f} N",
                f"{res_inc['Fs_max']:.2f} N", f"{res_inc['Fk']:.2f} N", res_inc['motion'],
                f"{res_belt['T_tight']:.2f} N", f"{res_belt['T_slack']:.2f} N",
                str(res_belt['ratio']),
                f"{res_jack['T_raise']:.3f} N·m", f"{res_jack['T_lower']:.3f} N·m",
                f"{res_jack['efficiency']}%",
                "YES" if res_jack["self_locking"] else "NO",
                mat_pair,
            ],
        }
        df_sum = pd.DataFrame(summary_data)
        st.dataframe(df_sum, use_container_width=True, hide_index=True)

        csv_bytes = df_sum.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Summary (CSV)",
            data=csv_bytes, file_name="friction_vision_summary.csv",
            mime="text/csv", use_container_width=True,
        )

    # ── Material reference table ──────────────────────────────────────────
    with st.expander("📚 Coefficient of Friction Reference Table"):
        mat_df = pd.DataFrame([
            {"Material Pair": k, "μs (Static)": v["mu_s"], "μk (Kinetic)": v["mu_k"],
             "φs (°)": round(np.degrees(np.arctan(v["mu_s"])), 1)}
            for k, v in MATERIAL_PAIRS.items() if k != "Custom"
        ])
        st.dataframe(mat_df, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"⚠️ Solver error: {e}")
    import traceback
    st.code(traceback.format_exc(), language="python")


# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:44px 0 32px 0; margin-top:36px;
  border-top:1px solid rgba(255,140,0,0.08);'>
  <div style='display:flex; align-items:center; justify-content:center;
    gap:18px; margin-bottom:20px;'>
    <div style='height:1px; width:60px;
      background:linear-gradient(90deg,transparent,rgba(255,140,0,0.25));'></div>
    <span style='color:rgba(255,140,0,0.30); font-size:10px;'>◆</span>
    <div style='height:1px; width:60px;
      background:linear-gradient(90deg,rgba(255,140,0,0.25),transparent);'></div>
  </div>
  <div style='display:inline-flex; align-items:center; gap:16px;
    font-family:Cinzel Decorative,serif; font-size:13px; font-weight:400;
    letter-spacing:0.18em; color:rgba(240,200,150,0.55);'>
    <span style='font-size:18px; color:rgba(255,140,0,0.50); line-height:1;'>✦</span>
    <span>Friction Vision — Engineering Mechanics Suite</span>
    <span style='font-size:18px; color:rgba(255,140,0,0.50); line-height:1;'>✦</span>
  </div>
</div>
""", unsafe_allow_html=True)
