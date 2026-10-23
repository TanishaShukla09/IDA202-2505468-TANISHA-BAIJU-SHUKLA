"""
FootLens Analytics - Player Injuries and Team Performance Dashboard
-------------------------------------------------------------------
Streamlit app for technical directors and sports managers.
Run locally:  streamlit run app.py

Code layout
    data_processing.py  -> load, clean, feature engineering (pandas / numpy)
    analysis.py         -> aggregations and statistical tests (pandas / scipy)
    app.py              -> layout, filters and interactive Plotly charts (this file)
"""
import html
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import analysis as an
from data_processing import PHASE_LABELS, SLOT_LABELS, get_clean_data

THEME_TOML = """[theme]
base = "light"
primaryColor = "#454832"
backgroundColor = "#E8DED0"
secondaryBackgroundColor = "#F5F0E8"
textColor = "#352A24"
font = "sans serif"
"""
_theme_file = Path.cwd() / ".streamlit" / "config.toml"
if not _theme_file.exists():
    try:
        _theme_file.parent.mkdir(exist_ok=True)
        _theme_file.write_text(THEME_TOML)
    except OSError:
        pass

st.set_page_config(page_title="Injury Impact | FootLens Analytics", page_icon="⚽", layout="wide")

# Refined editorial palette: olive, mocha, warm ivory and powder blue.
INK, DEEP, BLUE, SKY, TEAL, SLATE, CORAL = "#352A24", "#454832", "#91A8C4", "#8FA3B0", "#879078", "#88796B", "#B86F5E"
BLUE_SCALE = ["#E8DED0", "#91A8C4", "#91A8C4", "#8FA3B0", "#454832"]
GRID = "#D4C7B8"
PHASE_COLOURS = {"Before injury": TEAL, "Player absent": CORAL, "After return": BLUE}
RESULT_COLOURS = {"win": TEAL, "draw": "#A79581", "lose": CORAL}
SEASON_MONTHS = ["Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]
POSITION_GROUPS = ["Goalkeeper", "Defender", "Midfielder", "Forward"]

PITCH_W, PITCH_L = 68, 105
POSITION_XY = {
    "Goalkeeper": (34, 7), "Left Back": (9, 25), "Center Back": (34, 23), "Right Back": (59, 25),
    "Defensive Midfielder": (34, 41), "Central Midfielder": (34, 54), "Left Midfielder": (9, 56), "Right Midfielder": (59, 56),
    "Attacking Midfielder": (34, 69), "Left winger": (9, 83), "Right winger": (59, 83), "Center Forward": (34, 92),
}
POSITION_SHORT = {
    "Goalkeeper": "GK", "Left Back": "LB", "Center Back": "CB", "Right Back": "RB", "Defensive Midfielder": "DM",
    "Central Midfielder": "CM", "Left Midfielder": "LM", "Right Midfielder": "RM", "Attacking Midfielder": "AM",
    "Left winger": "LW", "Right winger": "RW", "Center Forward": "CF",
}
DROP_SCALE = [[0, "#91A8C4"], [0.5, "#F5F0E8"], [1, "#B86F5E"]]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Barlow:wght@400;500;600&display=swap');
    :root { color-scheme: light; }
    html, body, [class*="css"] { font-family: 'Barlow', sans-serif; }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: #E8DED0 !important; }
    [data-testid="stHeader"] { background: rgba(16,17,22,0.92) !important; }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child { background: #E2D6C7 !important; border-right: 1px solid #D4C7B8; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
    [data-testid="stWidgetLabel"] p, .stRadio label p, [data-testid="stExpander"] summary p { color: #352A24 !important; }
    h1, h2, h3, h4 { font-family: 'Barlow Condensed', sans-serif; letter-spacing: 0.3px; }
    .block-container { padding-top: 1.7rem; max-width: 1600px; }
    .hero { background:
        radial-gradient(circle at 88% 48%, transparent 65px, rgba(198,241,53,0.20) 65px, rgba(198,241,53,0.20) 67px, transparent 67px),
        linear-gradient(90deg, transparent calc(88% - 1px), rgba(198,241,53,0.24) calc(88% - 1px), rgba(198,241,53,0.24) calc(88% + 1px), transparent calc(88% + 1px)),
        linear-gradient(115deg, #454832 0%, #5A5D47 54%, #7A8065 100%);
        padding: 1.65rem 1.9rem; border: 1px solid #D4C7B8; border-left: 5px solid #454832; border-radius: 18px; margin-bottom: 1.2rem; box-shadow: 0 14px 32px rgba(0,0,0,0.18); }
    .stApp .hero h1 { color: #F7F7F8 !important; font-size: 2.65rem; margin: 0 0 0.35rem 0; }
    .stApp .hero p { color: #F5F0E8 !important; margin: 0; font-size: 1.02rem; max-width: 950px; }
    [data-testid="stMetric"] { background: #F5F0E8; border: 1px solid #D4C7B8; border-top: 3px solid #454832; border-radius: 14px; padding: 0.8rem 0.95rem; box-shadow: 0 8px 22px rgba(0,0,0,0.12); }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] div { font-family: 'Barlow Condensed', sans-serif; font-size: 2.1rem; color: #352A24 !important; }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] div { color: #76685E !important; font-size: 0.9rem; }
    [data-testid="stMetricDelta"] { color: #454832 !important; }
    button[data-baseweb="tab"] { background: transparent; border-radius: 9px 9px 0 0; }
    button[data-baseweb="tab"] p { color: #76685E !important; font-weight: 500; }
    button[data-baseweb="tab"][aria-selected="true"] { border-bottom: 3px solid #454832; }
    button[data-baseweb="tab"][aria-selected="true"] p { color: #454832 !important; font-weight: 700; }
    [data-testid="stPlotlyChart"] { background: #F5F0E8; border: 1px solid #D4C7B8; border-radius: 14px; padding: 6px; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid #D4C7B8; border-radius: 12px; overflow: hidden; }
    [data-testid="stExpander"] { background: #F5F0E8; border: 1px solid #D4C7B8; border-radius: 12px; }
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input { background: #EDE4D9 !important; color: #352A24 !important; border-color: #C4B5A4 !important; }
    [data-baseweb="select"] > div, [data-baseweb="popover"] { background: #EDE4D9 !important; border-color: #C4B5A4 !important; }
    .takeaway { background: #E6E0D4; border: 1px solid #C7C7AE; border-left: 4px solid #454832; border-radius: 10px; padding: 0.7rem 0.95rem; margin-top: 0.55rem; color: #454832; font-size: 0.95rem; line-height: 1.4; }
    .takeaway.tall { min-height: 5.6rem; }
    .fg { display: flex; flex-wrap: wrap; gap: 1.1rem; background: #F5F0E8; border: 1px solid #D4C7B8; border-radius: 12px; padding: 0.8rem 1rem; margin-bottom: 0.8rem; }
    .fgg { padding: 0.3rem 0.6rem; border-radius: 10px; }
    .fgg.absent { background: #E5DCD0; }
    .fgl { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: #454832; margin-bottom: 0.4rem; }
    .fgrow { display: flex; gap: 0.5rem; }
    .fgc { display: flex; flex-direction: column; align-items: center; width: 76px; }
    .fgb { width: 38px; height: 38px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; color: #E8DED0; }
    .fgb.W { background: #879078; } .fgb.D { background: #A79581; color: #E8DED0; } .fgb.L { background: #B86F5E; }
    .fgo { font-size: 0.72rem; color: #76685E; margin-top: 0.2rem; text-align: center; line-height: 1.1; }
    .fgr { font-size: 0.82rem; font-weight: 600; color: #352A24; }
    .question { font-family: 'Barlow Condensed', sans-serif; font-size: 1.3rem; font-weight: 700; margin-top: 0.8rem; color: #454832; }
    hr { border-color: #D4C7B8 !important; }
    @media (max-width: 760px) { .block-container { padding-top: 1rem; } .hero { padding: 1.2rem; } .stApp .hero h1 { font-size: 2rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Cleaning data...")
def load_everything(file_bytes=None):
    return get_clean_data(io.BytesIO(file_bytes) if file_bytes else None)


with st.sidebar.expander("Data source (optional)"):
    uploaded = st.file_uploader("Use a different CSV with the same columns", type="csv")
try:
    df, long_df, quality = load_everything(uploaded.getvalue() if uploaded else None)
except Exception as err:
    st.error(f"Could not load the data: {err}. Check that player_injuries_impact.csv is in the same folder as app.py (or in a data folder).")
    st.stop()


def fmt(x, nd=2, signed=False):
    """Format a number safely (handles NaN / None)."""
    if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def takeaway(text, tall=False):
    """Blue call-out under a chart that says what the chart means (the 'so what')."""
    st.markdown(f"<div class='takeaway{' tall' if tall else ''}'>{text}</div>", unsafe_allow_html=True)


def position_stats(data):
    """Injuries, average days out and average team drop for each pitch position, with pitch coordinates."""
    g = (data.groupby("position")
             .agg(injuries=("injury_id", "count"), avg_days=("days_out", "mean"), avg_drop=("performance_drop_index", "mean"))
             .reset_index())
    g["x"] = g["position"].map(lambda p: POSITION_XY.get(p, (np.nan, np.nan))[0])
    g["y"] = g["position"].map(lambda p: POSITION_XY.get(p, (np.nan, np.nan))[1])
    return g.dropna(subset=["x"])


def pitch_lines():
    """Football pitch markings as x/y lists (None separates the lines) in metres: 68 x 105."""
    xs, ys = [], []

    def add(points):
        for px_, py_ in points:
            xs.append(px_)
            ys.append(py_)
        xs.append(None)
        ys.append(None)

    add([(0, 0), (PITCH_W, 0), (PITCH_W, PITCH_L), (0, PITCH_L), (0, 0)])
    add([(0, PITCH_L / 2), (PITCH_W, PITCH_L / 2)])
    angles = np.linspace(0, 2 * np.pi, 80)
    add([(PITCH_W / 2 + 9.15 * np.cos(a), PITCH_L / 2 + 9.15 * np.sin(a)) for a in angles])
    for flip in (False, True):
        sign = (lambda v: PITCH_L - v) if flip else (lambda v: v)
        add([(13.84, sign(0)), (13.84, sign(16.5)), (54.16, sign(16.5)), (54.16, sign(0))])
        add([(24.84, sign(0)), (24.84, sign(5.5)), (43.16, sign(5.5)), (43.16, sign(0))])
    return xs, ys


def form_guide(rows):
    """Football-style form guide: W / D / L chips with opponent and player rating for each match around an injury."""
    groups = [("Before injury", "before_injury", ""), ("Player absent", "missed_match", " absent"), ("After return", "after_injury", "")]
    parts = ["<div class='fg'>"]
    for label, phase_key, extra in groups:
        chips = ""
        for r in rows[rows["phase"] == phase_key].itertuples():
            letter = {"win": "W", "draw": "D", "lose": "L"}.get(r.result, "-")
            rating = f"{r.rating:.1f}" if pd.notna(r.rating) else "&nbsp;"
            opponent = html.escape(str(r.opposition))[:14] if pd.notna(r.opposition) else ""
            chips += f"<div class='fgc'><div class='fgb {letter}'>{letter}</div><div class='fgo'>{opponent}</div><div class='fgr'>{rating}</div></div>"
        parts.append(f"<div class='fgg{extra}'><div class='fgl'>{label}</div><div class='fgrow'>{chips}</div></div>")
    parts.append("</div>")
    return "".join(parts)


def style(fig, height=380, legend=True):
    """Apply one consistent look to every Plotly figure."""
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=50, b=10), template="plotly_white", paper_bgcolor="#F5F0E8", plot_bgcolor="#F5F0E8",
        font=dict(family="Barlow, sans-serif", size=13, color=INK),
        title=dict(font=dict(family="Barlow Condensed, sans-serif", size=21, color=DEEP)),
        showlegend=legend, hoverlabel=dict(font_size=13, font_color="#352A24", bgcolor="#F5F0E8"),
        legend=dict(font=dict(color=INK), bgcolor="rgba(245,240,232,0.96)", bordercolor="#C4B5A4", borderwidth=1),
    )
    fig.update_xaxes(color=INK, gridcolor=GRID, linecolor="#B9AA99", zerolinecolor="#B9AA99")
    fig.update_yaxes(color=INK, gridcolor=GRID, linecolor="#B9AA99", zerolinecolor="#B9AA99")
    return fig


st.sidebar.header("⚽ Filters")
st.sidebar.caption("Every chart and number on every tab responds to these filters.")
clubs = sorted(df["team"].unique())
seasons = sorted(df["season"].unique())
types = sorted(df["injury_type"].unique())
sel_clubs = st.sidebar.multiselect("Club", clubs, default=clubs)
sel_seasons = st.sidebar.multiselect("Season", seasons, default=seasons)
sel_pos = st.sidebar.multiselect("Position group", POSITION_GROUPS, default=POSITION_GROUPS)
sel_types = st.sidebar.multiselect("Injury type", types, default=types)
age_lo, age_hi = int(df["age"].min()), int(df["age"].max())
sel_age = st.sidebar.slider("Player age", age_lo, age_hi, (age_lo, age_hi))

mask = (df["team"].isin(sel_clubs) & df["season"].isin(sel_seasons) & df["position_group"].isin(sel_pos)
        & df["injury_type"].isin(sel_types) & df["age"].between(*sel_age))
fdf = df[mask]
flong = long_df[long_df["injury_id"].isin(fdf["injury_id"])]

st.sidebar.download_button("Download filtered data (CSV)", fdf.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_injuries.csv", mime="text/csv")
st.sidebar.caption(f"Showing {len(fdf):,} of {len(df):,} injuries")

st.markdown(
    """
    <div class="hero">
      <h1>⚽ How do injuries change a club's results?</h1>
      <p>Premier League injuries from 2019/20 to 2023/24, compared with team results and player form before, during and after
      each absence. Built for technical directors planning rotation, training load and squad depth.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if fdf.empty:
    st.warning("No injuries match these filters. Widen the club, season, position, injury type or age filters in the sidebar.")
    st.stop()

absent = flong[flong["phase"] == "missed_match"]
win_rate = (absent["result"] == "win").mean() * 100 if len(absent) else np.nan
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("⚽ Injuries", f"{len(fdf):,}", help=f"{fdf['player'].nunique()} different players")
k2.metric("⏱️ Median days out", fmt(fdf["days_out"].median(), 0),
          help=f"Mean is {fmt(fdf['days_out'].mean(), 1)} days; a few long-term injuries pull the mean up.")
k3.metric("📉 Avg team drop index", fmt(fdf["performance_drop_index"].mean(), 2, signed=True),
          help="Goal difference per match before the injury minus goal difference while the player was absent. "
               "Above 0 means the team did worse without him.")
k4.metric("🏆 Win rate while absent", f"{fmt(win_rate, 0)}%", help="Share of missed matches the team won (up to 3 per injury).")
k5.metric("🔁 Avg rating change", fmt(fdf["rating_change"].mean(), 2, signed=True),
          help="Player match rating after returning minus before the injury.")

tab_over, tab_pat, tab_play, tab_3d, tab_ins, tab_about = st.tabs(
    ["🏟️ Overview", "📍 Pitch & patterns", "🩹 Player recovery", "🧊 3D explorer", "💡 Insights", "📋 About the data"])

with tab_over:
    st.subheader("Which injuries hurt the team most, and what happens while a player is out?")
    min_cases = st.slider("Minimum injuries per type", 3, 30, 10,
                          help="Types with only a few cases are mostly noise, so they are hidden.")
    c1, c2 = st.columns([3, 2])

    with c1:
        impact = an.injury_type_impact(fdf, min_cases).head(10)
        if impact.empty:
            st.info("No injury type has that many cases with the current filters. Lower the minimum.")
        else:
            impact = impact.assign(
                Effect=np.where(impact["avg_drop_index"] > 0, "Team did worse", "Team did better or the same"))
            fig = px.bar(impact.sort_values("avg_drop_index"), x="avg_drop_index", y="injury_type", orientation="h",
                         color="Effect", color_discrete_map={"Team did worse": CORAL, "Team did better or the same": BLUE},
                         hover_data={"cases": True, "avg_days_out": ":.0f", "avg_points_drop": ":.2f", "Effect": False},
                         labels={"avg_drop_index": "Average goal difference lost per match", "injury_type": "",
                                 "cases": "Injuries", "avg_days_out": "Avg days out", "avg_points_drop": "Points lost per match"},
                         title="Top 10 injury types by team performance drop")
            st.plotly_chart(style(fig, 440, legend=False), width="stretch", theme=None)
            worst = impact.iloc[0]
            takeaway(f"After {worst['injury_type']} injuries ({int(worst['cases'])} cases) teams lost {fmt(worst['avg_drop_index'])} goals of "
                     "goal difference per match. Red bars mean the team did worse without the player; blue bars mean it did no worse.", tall=True)

    with c2:
        phase = an.phase_performance(flong)
        fig = px.bar(phase, x="Phase", y="points_per_match", color="Phase", color_discrete_map=PHASE_COLOURS,
                     text="points_per_match", hover_data={"matches": True, "avg_gd": ":.2f", "Phase": False},
                     labels={"points_per_match": "Points per match", "matches": "Matches", "avg_gd": "Avg goal difference"},
                     title="Team points per match by phase")
        fig.update_traces(textposition="outside")
        fig.update_yaxes(range=[0, 3])
        st.plotly_chart(style(fig, 440, legend=False), width="stretch", theme=None)
        before, during = phase.iloc[0]["points_per_match"], phase.iloc[1]["points_per_match"]
        takeaway(f"Teams took {fmt(before)} points per match before the injury and {fmt(during)} while the player was absent. "
                 "A win is worth 3 points, a draw 1.", tall=True)

    c3, c4 = st.columns(2)
    with c3:
        rec = an.absence_record(flong).sort_values("Win %")
        fig = go.Figure()
        for col, colour in [("Win %", BLUE), ("Draw %", "#B7C9E2"), ("Loss %", CORAL)]:
            fig.add_bar(y=rec["Club"], x=rec[col], name=col.replace(" %", ""), orientation="h", marker_color=colour,
                        customdata=rec["Matches"], hovertemplate="%{y}: %{x:.1f}% (of %{customdata} matches)<extra>" + col + "</extra>")
        fig.update_layout(barmode="stack", title="Result record while a player was absent", xaxis_title="% of missed matches")
        fig = style(fig, 440)
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=70), legend=dict(orientation="h", y=-0.2, font=dict(color=INK)))
        st.plotly_chart(fig, width="stretch", theme=None)
        if len(rec):
            takeaway(f"{rec.iloc[-1]['Club']} won {fmt(rec.iloc[-1]['Win %'], 0)}% of matches with an injured player missing, "
                     f"{rec.iloc[0]['Club']} only {fmt(rec.iloc[0]['Win %'], 0)}%. Club strength matters as much as who is injured.", tall=True)
    with c4:
        cs = an.club_summary(fdf)
        fig = px.bar(cs.sort_values("injuries"), x="injuries", y="team", orientation="h", color="avg_drop_index",
                     color_continuous_scale=[BLUE, "#EEF5FD", CORAL], color_continuous_midpoint=0,
                     hover_data={"players_hit": True, "avg_days_out": ":.0f", "avg_drop_index": ":.2f"},
                     labels={"injuries": "Injuries recorded", "team": "", "avg_drop_index": "Avg drop index",
                             "players_hit": "Players hit", "avg_days_out": "Avg days out"},
                     title="Injuries per club (colour = team drop index)")
        fig = style(fig, 440)
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=70))
        st.plotly_chart(fig, width="stretch", theme=None)
        top_club = cs.iloc[0]
        takeaway(f"{top_club['team']} recorded the most injuries ({int(top_club['injuries'])}). "
                 "Red bars are clubs whose results suffered more when injured players were missing.", tall=True)

with tab_pat:
    st.subheader("Where on the pitch and when in the season do injuries happen?")
    ps = position_stats(fdf)
    pc1, pc2 = st.columns([5, 4])
    with pc1:
        fig = go.Figure()
        stripe = PITCH_L / 7
        fig.update_layout(shapes=[dict(type="rect", x0=0, x1=PITCH_W, y0=i * stripe, y1=(i + 1) * stripe, line_width=0, layer="below",
                                       fillcolor="#0D2B5E" if i % 2 == 0 else "#11356F") for i in range(7)])
        lx, ly = pitch_lines()
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color="#8DBCF7", width=1.6), hoverinfo="skip", showlegend=False))
        if not ps.empty:
            sizes = 14 + 17 * np.sqrt(ps["injuries"] / max(float(ps["injuries"].max()), 1.0))
            fig.add_trace(go.Scatter(
                x=ps["x"], y=ps["y"], mode="markers+text", text=ps["injuries"], textposition="middle center", textfont=dict(color="#101B3D", size=11, family="Barlow Condensed, sans-serif"),
                marker=dict(size=sizes, color=ps["avg_drop"], colorscale=DROP_SCALE, cmid=0, line=dict(color="#FFFFFF", width=2),
                            showscale=True, colorbar=dict(title=dict(text="Avg team drop", font=dict(color=INK)),
                                                          tickfont=dict(color=INK), len=0.55, thickness=12, x=1.02)),
                customdata=np.stack([ps["position"], ps["injuries"], ps["avg_days"].fillna(0), ps["avg_drop"].fillna(0)], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b><br>Injuries: %{customdata[1]}<br>Avg days out: %{customdata[2]:.0f}"
                              "<br>Avg team drop: %{customdata[3]:+.2f}<extra></extra>", showlegend=False))
            # Keep role labels in a dedicated trace, offset away from the bubble markers.
            label_offsets = {
                "Goalkeeper": -7.5, "Left Back": -7.5, "Center Back": -8.5, "Right Back": -7.5,
                "Defensive Midfielder": -8.0, "Central Midfielder": 10.0,
                "Left Midfielder": -8.0, "Right Midfielder": -8.0,
                "Attacking Midfielder": 9.0, "Left winger": -8.0, "Right winger": -8.0,
                "Center Forward": -8.0,
            }
            label_y = [row.y + label_offsets.get(row.position, -6.0) for row in ps.itertuples()]
            fig.add_trace(go.Scatter(x=ps["x"], y=label_y, mode="text", text=ps["position"],
                                     textfont=dict(color="#DCE9FB", size=10), hoverinfo="skip", showlegend=False))
        fig.add_annotation(x=PITCH_W / 2, y=-2.2, text="Attacking direction", showarrow=False, font=dict(color="#8DBCF7", size=12))
        fig.update_layout(title="Injury map: bubble size = injuries, colour = team drop when he is out",
                          xaxis=dict(visible=False, range=[-3, PITCH_W + 3], constrain="domain"),
                          yaxis=dict(visible=False, range=[-8, PITCH_L + 4], scaleanchor="x", scaleratio=1),
                          margin=dict(l=12, r=22, t=58, b=24), autosize=True)
        fig = style(fig, 650, legend=False)
        fig.update_layout(plot_bgcolor="#F5F0E8", paper_bgcolor="#F5F0E8")
        st.plotly_chart(fig, width="stretch", theme=None)
    with pc2:
        if ps.empty:
            st.info("No positions to show with these filters.")
        else:
            by_pos = ps.sort_values("injuries")
            fig = px.bar(by_pos, x="injuries", y="position", orientation="h", color_discrete_sequence=[BLUE],
                         labels={"injuries": "Injuries", "position": ""}, title="Injuries by position")
            st.plotly_chart(style(fig, 330, legend=False), width="stretch", theme=None)
            solid = ps[ps["injuries"] >= 8].sort_values("avg_drop")
            if solid.empty:
                st.info("No position has 8 or more injuries with these filters.")
            else:
                solid = solid.assign(Effect=np.where(solid["avg_drop"] > 0, "Team did worse", "No worse"))
                fig = px.bar(solid, x="avg_drop", y="position", orientation="h", color="Effect",
                             color_discrete_map={"Team did worse": CORAL, "No worse": BLUE},
                             labels={"avg_drop": "Avg goal difference lost per match", "position": ""},
                             title="Team drop by position (8+ injuries)")
                st.plotly_chart(style(fig, 330, legend=False), width="stretch", theme=None)
    if not ps.empty:
        most_pos = ps.sort_values("injuries").iloc[-1]
        solid_all = ps[ps["injuries"] >= 8].sort_values("avg_drop")
        extra = (f" The biggest drop in results follows {solid_all.iloc[-1]['position']} injuries "
                 f"({fmt(solid_all.iloc[-1]['avg_drop'], 2, True)} goals per match)." if len(solid_all) else "")
        takeaway(f"{most_pos['position']} is the most injured position ({int(most_pos['injuries'])} injuries).{extra} "
                 "Bubbles near the centre of the pitch are the busiest because several players share those roles.")

    heat = an.month_club_heatmap(fdf, SEASON_MONTHS)
    fig = px.imshow(heat, text_auto=True, aspect="auto", color_continuous_scale=BLUE_SCALE,
                    labels=dict(x="Month of injury", y="", color="Injuries"), title="Injury frequency by club and month (season order)")
    st.plotly_chart(style(fig, 380, legend=False), width="stretch", theme=None)
    month_totals = heat.sum()
    peak_cell = heat.stack().idxmax()
    takeaway(f"Injuries peak in {month_totals.idxmax()} ({int(month_totals.max())} injuries) and are lowest in "
             f"{month_totals.idxmin()} ({int(month_totals.min())}). The single busiest cell is {peak_cell[0]} in {peak_cell[1]} "
             f"({int(heat.stack().max())} injuries).")

    c1, c2 = st.columns(2)
    with c1:
        sc = fdf.dropna(subset=["age", "performance_drop_index"])
        fig = px.scatter(sc, x="age", y="performance_drop_index", color="position_group", opacity=0.65,
                         color_discrete_sequence=["#5B5FEF", "#18B6A4", "#B86F5E", "#F6B94A"],
                         hover_data={"player": True, "team": True, "injury_raw": True, "season": True},
                         labels={"age": "Player age", "performance_drop_index": "Team performance drop index",
                                 "position_group": "Position", "injury_raw": "Injury"},
                         title="Player age vs team performance drop")
        corr = an.age_correlation(fdf)
        if not np.isnan(corr["r"]):
            xs = np.array([sc["age"].min(), sc["age"].max()])
            fig.add_trace(go.Scatter(x=xs, y=corr["slope"] * xs + corr["intercept"], mode="lines", name="Trend line",
                                     line=dict(color=INK, dash="dash")))
        fig.add_hline(y=0, line_color="#B7C9E2", line_width=1)
        st.plotly_chart(style(fig, 440), width="stretch", theme=None)
        takeaway(f"Correlation r = {fmt(corr['r'])} (n = {corr['n']}), {an.significance_label(corr['p'])}. "
                 "A value near 0 means a player's age tells us almost nothing about how much the team suffers.", tall=True)
    with c2:
        fig = px.box(fdf.dropna(subset=["performance_drop_index"]), x="star_player", y="performance_drop_index",
                     color="star_player", points="all", color_discrete_sequence=[DEEP, TEAL],
                     category_orders={"star_player": ["Star (FIFA 82+)", "Squad player"]},
                     hover_data={"player": True, "team": True, "star_player": False},
                     labels={"star_player": "", "performance_drop_index": "Team performance drop index"},
                     title="Star players vs squad players")
        fig.update_traces(boxmean=True, jitter=0.5, marker_opacity=0.4)
        st.plotly_chart(style(fig, 440, legend=False), width="stretch", theme=None)
        stt = an.star_vs_squad_test(fdf)
        if "p" in stt:
            takeaway(f"Average drop: stars {fmt(stt['mean_star'], 2, True)} vs squad players {fmt(stt['mean_squad'], 2, True)} "
                     f"(Welch t-test p = {fmt(stt['p'], 2)}, {an.significance_label(stt['p'])}). Dashed line = mean.", tall=True)
        else:
            takeaway("Not enough star or squad injuries with these filters for a comparison.", tall=True)

    days = fdf.dropna(subset=["days_out"])
    order = days.groupby("injury_type")["days_out"].median().sort_values(ascending=False).index.tolist()
    fig = px.box(days, x="injury_type", y="days_out", log_y=True, category_orders={"injury_type": order},
                 color_discrete_sequence=[BLUE], hover_data={"player": True, "team": True},
                 labels={"injury_type": "", "days_out": "Days out (log scale)"}, title="How long each injury type keeps players out")
    fig.update_xaxes(tickangle=-35)
    st.plotly_chart(style(fig, 400, legend=False), width="stretch", theme=None)
    takeaway(f"Median absence is longest for {order[0]} injuries and shortest for {order[-1]}. "
             "The log scale keeps a handful of very long layoffs from squashing the rest.")

with tab_play:
    st.subheader("How do players and teams perform around an injury?")
    view = st.radio("Timeline view", ["One player's injury", "Average of all filtered injuries"], horizontal=True)
    slot_order = list(SLOT_LABELS.values())
    guide_html = ""

    if view == "One player's injury":
        counts = fdf.groupby("player")["injury_id"].count().sort_values(ascending=False)
        player = st.selectbox("Player", counts.index.tolist(), format_func=lambda n: f"{n} ({counts[n]} injuries)")
        p_inj = fdf[fdf["player"] == player].sort_values("injury_date")
        labels = {r.injury_id: f"{r.injury_date:%d %b %Y} - {r.injury_raw}" + (f" ({int(r.days_out)} days)" if pd.notna(r.days_out) else "")
                  for r in p_inj.itertuples()}
        inj_id = st.selectbox("Injury", list(labels), format_func=lambda i: labels[i])
        guide_html = form_guide(long_df[long_df["injury_id"] == inj_id])
        tl = long_df[long_df["injury_id"] == inj_id].set_index("slot_label").reindex(slot_order)
        title = f"{player}: match ratings and team goal difference around the injury"
    else:
        tl = flong.groupby("slot_label").agg(rating=("rating", "mean"), gd=("gd", "mean"), result=("result", "first")).reindex(slot_order)
        tl["result"] = np.where(tl["gd"] > 0.05, "win", np.where(tl["gd"] < -0.05, "lose", "draw"))
        title = "Average player rating and team goal difference around an injury"

    if guide_html:
        st.markdown("#### Form guide: the nine matches around this injury")
        st.markdown(guide_html, unsafe_allow_html=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.08,
                        subplot_titles=("Player match rating (substitute appearances included)", "Team goal difference in the match"))
    fig.add_trace(go.Scatter(x=slot_order, y=tl["rating"], mode="lines+markers", connectgaps=False, name="Player rating",
                             line=dict(color=INK, width=2), marker=dict(size=9, color=INK),
                             hovertemplate="%{x}<br>Rating %{y:.1f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=slot_order, y=tl["gd"], name="Goal difference",
                         marker_color=[RESULT_COLOURS.get(r, "#C5D3E3") if pd.notna(r) else "#C5D3E3" for r in tl["result"]],
                         hovertemplate="%{x}<br>Goal difference %{y:+.1f}<extra></extra>"), row=2, col=1)
    fig.add_vrect(x0=2.5, x1=5.5, fillcolor=SKY, opacity=0.25, line_width=0, row="all", col=1)
    fig.add_annotation(x=4, y=1.0, xref="x", yref="paper", text="Player absent", showarrow=False, font=dict(color=DEEP, size=13))
    fig.update_yaxes(title_text="Rating", row=1, col=1)
    fig.update_yaxes(title_text="Goal difference", row=2, col=1)
    fig.update_xaxes(categoryorder="array", categoryarray=slot_order)
    fig.update_layout(title=title)
    st.plotly_chart(style(fig, 520, legend=False), width="stretch", theme=None)
    takeaway("Match 'Before -1' is the last game before the injury and 'After +1' the first game after returning. "
             "No rating is recorded while the player is absent, so that part of the line is empty. "
             "Blue bars are wins, grey draws and red defeats.")

    st.markdown("#### Comeback leaderboard")
    max_inj = int(fdf.groupby("player")["injury_id"].count().max())
    lc1, lc2 = st.columns(2)
    top_n = lc1.slider("Players to show", 5, 20, 10)
    min_inj = lc2.slider("Minimum injuries per player", 1, max(2, min(5, max_inj)), 1,
                         help="Raise this to rank only players with several injuries, which is less affected by one lucky or unlucky match.")
    best = an.comeback_leaderboard(fdf, top_n, min_inj)
    worst = an.comeback_leaderboard(fdf, top_n, min_inj, worst=True)
    if best.empty:
        st.info("No players have a rating both before and after an injury with these filters.")
    else:
        best.insert(0, "Rank", range(1, len(best) + 1))
        cfg = {"Rank": st.column_config.NumberColumn("Rank", width="small"),
               "player": "Player", "team": "Club", "injuries": "Injuries",
               "rating_before": st.column_config.NumberColumn("Rating before", format="%.2f"),
               "rating_after": st.column_config.NumberColumn("Rating after", format="%.2f"),
               "rating_change": st.column_config.NumberColumn("Change", format="%+.2f")}
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Strongest comebacks** (rating improved most)")
            st.dataframe(best, hide_index=True, column_config=cfg, width="stretch")
        with t2:
            st.markdown("**Biggest declines** (rating fell most)")
            worst.insert(0, "Rank", range(1, len(worst) + 1))
            st.dataframe(worst, hide_index=True, column_config=cfg, width="stretch")
        fig = px.bar(best.sort_values("rating_change"), x="rating_change", y="player", orientation="h",
                     color_discrete_sequence=[BLUE], hover_data={"team": True, "rating_before": ":.2f", "rating_after": ":.2f"},
                     labels={"rating_change": "Rating change after return", "player": ""}, title="Comeback players ranked by rating improvement")
        st.plotly_chart(style(fig, 380, legend=False), width="stretch", theme=None)
        takeaway("Ratings come from at most three matches before and after each injury, so a single big game can move a player a long way. "
                 "Raise the minimum injuries slider to see steadier comebacks.")

    with st.expander("Player summary across injury phases (before, during absence, after return)"):
        st.write("Average team goal difference and player rating for every player, grouped across all of their injuries. Click a column heading to sort.")
        st.dataframe(an.player_phase_summary(fdf), hide_index=True, width="stretch")
    with st.expander("Most injured players"):
        st.dataframe(an.most_injured_players(fdf, 15), hide_index=True, width="stretch")

with tab_3d:
    st.subheader("3D explorer: drag to rotate, scroll to zoom, double-click to reset")
    st.caption("Rotate, zoom and hover to explore the data. Spherical markers encode individual injuries; their size and colour represent the selected variables. All views respond to the sidebar filters.")
    axis_style = dict(backgroundcolor="#F5F0E8", gridcolor="#D4C7B8", showbackground=True, color=INK)

    st.markdown("#### 1. Injury cube: age, time out and team impact")
    colour_choice = st.radio("Colour points by", ["Position group", "Injury type", "Star status"], horizontal=True)
    colour_col = {"Position group": "position_group", "Injury type": "injury_type", "Star status": "star_player"}[colour_choice]
    cube = fdf.dropna(subset=["age", "days_out", "performance_drop_index"]).assign(days_out=lambda x: x["days_out"].clip(lower=1))
    cube = cube.assign(fifa_size=(cube["fifa_rating"].fillna(cube["fifa_rating"].median()) - 60).clip(lower=1))
    if len(cube) < 3:
        st.info("Not enough injuries with complete data for a 3D cube with these filters.")
    else:
        fig = px.scatter_3d(cube, x="age", y="days_out", z="performance_drop_index", color=colour_col, size="fifa_size",
                            size_max=14, opacity=0.8, log_y=True,
                            color_discrete_sequence=["#5B5FEF", "#18B6A4", "#91A8C4", "#B86F5E", "#F6B94A", "#9B72CF",
                                                     "#263A9B", "#79D7C7", "#F28E5B", "#8B9BB8", "#E879F9", "#38BDF8", "#A3E635", "#7C8DB5", "#B86F5E"],
                            hover_data={"player": True, "team": True, "injury_raw": True, "fifa_size": False},
                            labels={"age": "Age", "days_out": "Days out (log)", "performance_drop_index": "Team drop index",
                                    "position_group": "Position", "injury_type": "Injury type", "star_player": "Status",
                                    "injury_raw": "Injury"},
                            title="Each point is one injury (bigger point = higher FIFA rating)")
        fig.update_layout(scene=dict(xaxis=axis_style, yaxis=axis_style, zaxis=axis_style,
                                     camera=dict(eye=dict(x=1.6, y=1.6, z=0.9))),
                          legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(style(fig, 620), width="stretch", theme=None)
        takeaway("Look for points high on the vertical axis (team did worse without the player). Rotate the cube to check whether "
                 "older players or longer layoffs sit higher. In this data they mostly do not, which matches the weak correlation in the Insights tab.")

    st.markdown("#### 2. Injury landscape: clubs and months as a 3D surface")
    heat3d = an.month_club_heatmap(fdf, SEASON_MONTHS)
    if heat3d.shape[0] < 2:
        st.info("Select at least two clubs to draw the 3D surface.")
    else:
        fig = go.Figure(go.Surface(z=heat3d.values, x=SEASON_MONTHS, y=heat3d.index.tolist(), colorscale=[[0, "#263A9B"], [0.25, "#5B5FEF"], [0.55, "#18B6A4"], [0.8, "#91A8C4"], [1, "#F6B94A"]],
                                   contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
                                   hovertemplate="Club: %{y}<br>Month: %{x}<br>Injuries: %{z}<extra></extra>",
                                   colorbar=dict(title="Injuries")))
        fig.update_layout(title="Peaks show injury hot spots by club and month",
                          scene=dict(xaxis=dict(title="Month (season order)", **axis_style), yaxis=dict(title="Club", **axis_style),
                                     zaxis=dict(title="Injuries", **axis_style), camera=dict(eye=dict(x=1.7, y=-1.6, z=0.9))))
        st.plotly_chart(style(fig, 560, legend=False), width="stretch", theme=None)
        peak = heat3d.stack().idxmax()
        takeaway(f"The tallest peak is {peak[0]} in {peak[1]} ({int(heat3d.stack().max())} injuries). "
                 "Drag the surface to look at it from the side and compare the winter months with the rest of the season.")

    st.markdown("#### 3. Recovery cube: rating before, rating after and time out")
    rec3d = fdf.dropna(subset=["rating_before", "rating_after", "days_out"]).assign(days_out=lambda x: x["days_out"].clip(lower=1))
    if len(rec3d) < 3:
        st.info("Not enough injuries with ratings before and after for this chart with the current filters.")
    else:
        fig = px.scatter_3d(rec3d, x="rating_before", y="rating_after", z="days_out", color="rating_change", log_z=True,
                            color_continuous_scale=["#B86F5E", "#F5F0E8", "#18B6A4"], color_continuous_midpoint=0, opacity=0.85,
                            hover_data={"player": True, "team": True, "injury_raw": True},
                            labels={"rating_before": "Rating before", "rating_after": "Rating after", "days_out": "Days out (log)",
                                    "rating_change": "Rating change", "injury_raw": "Injury"},
                            title="Points on the diagonal returned at the same level")
        fig.update_layout(scene=dict(xaxis=axis_style, yaxis=axis_style, zaxis=axis_style,
                                     camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8))))
        st.plotly_chart(style(fig, 580, legend=False), width="stretch", theme=None)
        takeaway("Blue points improved after returning and red points declined. Most points sit near the diagonal where rating before "
                 "equals rating after, so a longer layoff does not clearly lead to a worse return.")

    st.markdown("#### 4. Pitch in 3D: injuries by position")
    ps3 = position_stats(fdf)
    if ps3.empty:
        st.info("No positions to show with these filters.")
    else:
        lx, ly = pitch_lines()
        fig = go.Figure()
        fig.add_trace(go.Mesh3d(x=[0, PITCH_W, PITCH_W, 0], y=[0, 0, PITCH_L, PITCH_L], z=[0, 0, 0, 0], i=[0, 0], j=[1, 2], k=[2, 3],
                                color="#0F2F66", opacity=1, flatshading=True, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=[0] * len(lx), mode="lines", line=dict(color=SKY, width=4), hoverinfo="skip", showlegend=False))
        stem_x, stem_y, stem_z = [], [], []
        for r in ps3.itertuples():
            stem_x += [r.x, r.x, None]
            stem_y += [r.y, r.y, None]
            stem_z += [0, r.injuries, None]
        fig.add_trace(go.Scatter3d(x=stem_x, y=stem_y, z=stem_z, mode="lines", line=dict(color=BLUE, width=8), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter3d(
            x=ps3["x"], y=ps3["y"], z=ps3["injuries"], mode="markers+text", text=ps3["position"].map(POSITION_SHORT),
            textposition="top center", textfont=dict(color=INK, size=13),
            marker=dict(size=10, color=ps3["avg_drop"], colorscale=DROP_SCALE, cmid=0, line=dict(color=INK, width=1), showscale=True,
                        colorbar=dict(title=dict(text="Avg team drop", font=dict(color=INK)), tickfont=dict(color=INK), len=0.6, thickness=12)),
            customdata=np.stack([ps3["position"], ps3["injuries"], ps3["avg_days"].fillna(0), ps3["avg_drop"].fillna(0)], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>Injuries: %{customdata[1]}<br>Avg days out: %{customdata[2]:.0f}"
                          "<br>Avg team drop: %{customdata[3]:+.2f}<extra></extra>", showlegend=False))
        fig.update_layout(title="Taller stems mean more injuries in that position (colour = team drop)",
                          scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                                     zaxis=dict(title="Injuries", showbackground=False, gridcolor="#D4C7B8", color=INK),
                                     aspectmode="manual", aspectratio=dict(x=0.6, y=1, z=0.55),
                                     camera=dict(eye=dict(x=1.5, y=-1.7, z=0.9))))
        st.plotly_chart(style(fig, 640, legend=False), width="stretch", theme=None)
        tallest = ps3.sort_values("injuries").iloc[-1]
        takeaway(f"The tallest stem is {tallest['position']} with {int(tallest['injuries'])} injuries. Rotate the pitch to compare defence, "
                 "midfield and attack, and hover over a stem top for days out and team drop.")


def build_insights(data: pd.DataFrame, match_data: pd.DataFrame) -> list[tuple[str, str]]:
    """Plain-English answers to the six research questions, computed from the filtered data."""
    out = []
    imp = an.injury_type_impact(data, 10)
    drop = an.team_drop_test(data)
    if len(imp) and "mean" in drop:
        top, low = imp.iloc[0], imp.iloc[-1]
        txt = (f"Among injury types with at least 10 cases, **{top['injury_type']}** injuries were followed by the largest drop "
               f"({fmt(top['avg_drop_index'], 2, True)} goals of goal difference per match, {int(top['cases'])} cases), while after "
               f"**{low['injury_type']}** injuries teams did {fmt(low['avg_drop_index'], 2, True)} (no worse). Across all filtered injuries the average drop is "
               f"{fmt(drop['mean'], 2, True)} (95% CI {fmt(drop['ci_low'], 2)} to {fmt(drop['ci_high'], 2)}), which is "
               f"{an.significance_label(drop['p'])}. Individual injury types differ, but with this sample size the average team effect is small.")
    else:
        txt = "Not enough injuries with these filters to compare injury types (needs 10 or more per type)."
    out.append(("1. Which injuries led to the biggest team performance drop?", txt))
    rec = an.absence_record(match_data)
    ph = an.phase_performance(match_data)
    if len(rec):
        b, w = rec.sort_values("Win %").iloc[-1], rec.sort_values("Win %").iloc[0]
        txt = (f"Teams collected {fmt(ph.iloc[0]['points_per_match'])} points per match before the injury, {fmt(ph.iloc[1]['points_per_match'])} "
               f"while the player was absent and {fmt(ph.iloc[2]['points_per_match'])} after return. By club, {b['Club']} won {fmt(b['Win %'], 0)}% "
               f"of matches without the injured player, versus {fmt(w['Win %'], 0)}% for {w['Club']}. The gap between clubs is bigger than the "
               f"gap between phases, so overall club strength matters more than any single absence.")
    else:
        txt = "No missed matches with these filters."
    out.append(("2. What was the team's win/loss record during player absence?", txt))
    pt = an.paired_rating_test(data)
    best = an.comeback_leaderboard(data, 1, 1)
    if "mean_diff" in pt:
        txt = (f"Across {pt['n']} injuries with ratings before and after, the average rating change is {fmt(pt['mean_diff'], 2, True)} "
               f"(95% CI {fmt(pt['ci_low'], 2)} to {fmt(pt['ci_high'], 2)}; paired t-test p = {fmt(pt['p'], 2)}, {an.significance_label(pt['p'])}). "
               f"In other words, players return to roughly the same level on average. ")
        if len(best):
            txt += (f"The strongest individual comeback is **{best.iloc[0]['player']}** ({fmt(best.iloc[0]['rating_before'], 1)} to "
                    f"{fmt(best.iloc[0]['rating_after'], 1)}), a reminder that single-injury results are noisy.")
    else:
        txt = "Not enough injuries with ratings before and after to test recovery with these filters."
    out.append(("3. How did individual players perform after recovery?", txt))
    heat = an.month_club_heatmap(data, SEASON_MONTHS)
    if heat.values.sum() > 0:
        mt = heat.sum()
        cell = heat.stack().idxmax()
        txt = (f"Injuries are most frequent in **{mt.idxmax()}** ({int(mt.max())}) and least frequent in {mt.idxmin()} ({int(mt.min())}). "
               f"The busiest club-month is **{cell[0]} in {cell[1]}** ({int(heat.stack().max())} injuries). The winter peak lines up with the busiest "
               f"stretch of the fixture calendar, and published research on professional football links short recovery between matches to more muscle injuries.")
    else:
        txt = "No injuries with these filters."
    out.append(("4. Are there specific months or clubs with frequent injury clusters?", txt))
    cs = an.club_summary(data)
    if len(cs):
        most = cs.iloc[0]
        big = cs[cs["injuries"] >= 20].sort_values("avg_drop_index", ascending=False)
        txt = f"**{most['team']}** record the most injuries ({int(most['injuries'])}). "
        if len(big):
            h = big.iloc[0]
            txt += (f"In terms of results, **{h['team']}** suffer most: their goal difference drops by {fmt(h['avg_drop_index'], 2, True)} per match "
                    f"when an injured player is missing and injuries keep players out for {fmt(h['avg_days_out'], 0)} days on average. ")
        txt += "Raw injury counts and real impact on results are different rankings, and squad depth is the likely reason."
    else:
        txt = "No clubs with these filters."
    out.append(("5. Which clubs suffer most due to injuries?", txt))
    st_t, corr = an.star_vs_squad_test(data), an.age_correlation(data)
    if "p" in st_t and not np.isnan(corr["r"]):
        txt = (f"Star players (FIFA 82+) are followed by an average drop of {fmt(st_t['mean_star'], 2, True)} versus {fmt(st_t['mean_squad'], 2, True)} "
               f"for other players (Welch t-test p = {fmt(st_t['p'], 2)}, {an.significance_label(st_t['p'])}). Age and drop index have a correlation of "
               f"r = {fmt(corr['r'])} ({an.significance_label(corr['p'])}). Neither a player's rating nor age is a reliable predictor of how much the team suffers.")
    else:
        txt = "Not enough data with these filters to compare star players or test age."
    out.append(("6. Do star players or older players hurt the team more when injured?", txt))
    return out


with tab_ins:
    st.subheader("What the data says")
    st.write("These answers are recalculated from your current filters. The statistical tests use a 5% significance level.")
    for q, a in build_insights(fdf, flong):
        st.markdown(f"<div class='question'>{q}</div>", unsafe_allow_html=True)
        st.write(a)

    st.markdown("#### What this means for a technical director (full dataset)")
    full_heat = an.month_club_heatmap(df, SEASON_MONTHS)
    full_cs = an.club_summary(df)
    heavy = full_cs[full_cs["injuries"] >= 20].sort_values("avg_drop_index", ascending=False).iloc[0]
    long_types = an.injury_type_impact(df, 10).sort_values("avg_days_out", ascending=False).iloc[0]
    st.markdown(
        f"- **Plan rotation before the winter peak.** {full_heat.sum().idxmax()} has the most injuries, so manage minutes and recovery days in the weeks before it.\n"
        f"- **Invest in squad depth, not only star signings.** Star players are not hit harder in results, but clubs such as {heavy['team']} lose "
        f"{fmt(heavy['avg_drop_index'], 2)} goals of goal difference per match on average when a player is missing.\n"
        f"- **Budget for long layoffs.** {long_types['injury_type']} injuries keep players out for {fmt(long_types['avg_days_out'], 0)} days on average, so cover is needed early.\n"
        f"- **Do not rush judgement on returning players.** Ratings after return match ratings before on average, so there is no evidence of a lasting dip.")

    st.markdown("#### Limitations")
    st.markdown(
        "- Each injury has at most three matches before, during and after, so averages are noisy and many results are not statistically significant.\n"
        "- Only eight clubs are included, and results depend on opponent strength, which the data does not control for.\n"
        "- Player ratings are missing for some appearances, and ratings are not recorded while a player is absent.\n"
        "- The data shows association, not proof that an injury caused a change in results.")

with tab_about:
    st.subheader("About the data and how it was prepared")
    st.write("The source file has one row per injury (656 injuries, 224 players, 8 clubs, 5 seasons) and 42 columns describing "
             "the three matches before the injury, the three matches missed and the three matches after returning.")
    st.markdown("#### Cleaning steps and what they found")
    st.dataframe(quality, hide_index=True, width="stretch")

    st.markdown("#### Columns created")
    st.dataframe(pd.DataFrame([
        ("days_out", "Return date minus injury date, in days"),
        ("rating_before / rating_after", "Mean player rating over the available matches before / after the injury"),
        ("rating_change", "rating_after minus rating_before (above 0 = improved)"),
        ("performance_drop_index", "Mean team goal difference before the injury minus mean goal difference while absent (above 0 = team did worse)"),
        ("points_drop", "Mean points per match before minus while absent (win 3, draw 1, loss 0)"),
        ("injury_type", "About 100 raw labels grouped into 15 categories"),
        ("position_group", "Goalkeeper, Defender, Midfielder or Forward"),
        ("star_player", "FIFA rating of 82 or higher"),
        ("injury_month / season", "Taken from the injury date and the season column"),
    ], columns=["Column", "Meaning"]), hide_index=True, width="stretch")

    st.markdown("#### Background research")
    st.markdown(
        "- Hägglund et al. (2013), *Injuries affect team performance negatively in professional football: an 11-year follow-up of the UEFA Champions League injury study*, British Journal of Sports Medicine. "
        "Found that injury burden and player availability were linked to team results, which motivated the team drop index.\n"
        "- *Muscle injury rates in professional football increase with fixture congestion*, British Journal of Sports Medicine (2013). Motivated the month-by-club view.\n"
        "- Streamlit and Plotly football dashboard examples listed in the assignment brief guided the layout.")
    with st.expander("Preview the cleaned data"):
        st.dataframe(df.drop(columns=[c for c in df.columns if c.startswith("Match")]).head(50), hide_index=True)