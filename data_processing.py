"""
data_processing.py
------------------
Loading, cleaning and feature engineering for the Player Injuries Impact dataset.

Pipeline (each step is its own function so it can be tested and explained):
    load_data()          -> raw DataFrame (everything read as text)
    clean_data()         -> typed, standardised DataFrame (one row per injury)
    engineer_features()  -> adds analytical columns (rating change, drop index, ...)
    to_long_format()     -> one row per match, used for the player timeline chart
    repair_return_dates() -> fixes year typos in return dates
    resolve_result_gd_conflicts() -> logic check between result and goal difference
    missing_value_report() -> missing values per column group and how they are handled
    run_quality_checks() -> summary table of issues found, shown on the About tab
    get_clean_data()     -> convenience wrapper used by the Streamlit app
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).parent
DATA_NAME = "player_injuries_impact.csv"


def find_data_file():
    """Look for the dataset next to this file, in a data folder, or anywhere in the project (any CSV with 'injur' in its name)."""
    for candidate in (_HERE / "data" / DATA_NAME, _HERE / DATA_NAME):
        if candidate.exists():
            return candidate
    for candidate in sorted(_HERE.rglob("*.csv")):
        if "injur" in candidate.name.lower():
            return candidate
    return None


DATA_PATH = find_data_file() or (_HERE / "data" / DATA_NAME)

NA_TOKENS = ["N.A.", "N.A", "n.a.", "NA", "N/A", ""]   # text used for "missing" in the raw file
RESULT_POINTS = {"win": 3, "draw": 1, "lose": 0}        # league points for a result
PHASES = ["before_injury", "missed_match", "after_injury"]
PHASE_LABELS = {"before_injury": "Before injury", "missed_match": "Player absent", "after_injury": "After return"}
MATCH_NUMBERS = [1, 2, 3]
SLOT_LABELS = {0: "Before -3", 1: "Before -2", 2: "Before -1", 3: "Absent 1", 4: "Absent 2",
               5: "Absent 3", 6: "After +1", 7: "After +2", 8: "After +3"}
SEVERITY_LABELS = ["Slight (0-1 d)", "Minimal (2-3 d)", "Mild (4-7 d)", "Moderate (8-28 d)", "Severe (29+ d)"]
VALID_YEARS = (2015, 2026)     # any date outside this range is a typing error (the file contains 'Mar 7, 0202')
STAR_THRESHOLD = 82            # FIFA rating at or above this = "star player" (roughly the top fifth)

# Ordered rules: the FIRST pattern that matches decides the category.
INJURY_RULES = [
    ("Illness / COVID-19", r"\b(ill|coronavirus|quarantine|virus|infection|tonsillitis|cold|fever|malaria|bronchitis|shingles|depression)\b"),
    ("Hamstring", r"hamstring"),
    ("Knee", r"knee|cruciate|meniscus"),
    ("Ankle", r"ankle"),
    ("Calf", r"calf"),
    ("Groin / adductor", r"groin|adductor|pubic"),
    ("Muscle / thigh", r"muscle|muscular|thigh"),
    ("Tendon / ligament", r"achilles|archilles|tendon|ligament"),
    ("Back / hip", r"\bhip\b|back|ack injury"),
    ("Foot / leg", r"foot|toe|\bleg\b|shin"),
    ("Head / neck", r"head|concussion|facial|neck"),
    ("Upper body", r"shoulder|chest|hand|\barm\b|elbow|abdominal"),
    ("Fracture / bone", r"broken|fracture|bone|metatarsal|fibula"),
    ("Knock / bruise", r"knock|bruise"),
    ("Unknown / fitness", r"unknown|fitness"),
]

POSITION_GROUPS = {
    "Goalkeeper": "Goalkeeper",
    "Center Back": "Defender", "Left Back": "Defender", "Right Back": "Defender",
    "Defensive Midfielder": "Midfielder", "Central Midfielder": "Midfielder",
    "Attacking Midfielder": "Midfielder", "Left Midfielder": "Midfielder", "Right Midfielder": "Midfielder",
    "Left winger": "Forward", "Right winger": "Forward", "Center Forward": "Forward",
}

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load_data(path=None) -> pd.DataFrame:
    """Read the CSV as text so that tokens such as 'N.A.' and '6.1(S)' are not silently coerced.
    `path` can be a file path or an open file object; None means the default dataset in the project."""
    if path is None:
        path = find_data_file()
        if path is None:
            top = sorted(p.name for p in _HERE.iterdir())
            data_dir = _HERE / "data"
            inside = sorted(p.name for p in data_dir.iterdir()) if data_dir.is_dir() else "(no data folder)"
            raise FileNotFoundError(
                f"{DATA_NAME} was not found. Files next to app.py: {top}. Files in data/: {inside}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def parse_date(value: str) -> pd.Timestamp:
    """Parse the dataset's mixed date formats ('Nov 9, 2019', 'Oct 15,2022', 'January 4, 2023').
    'Present' (still injured) and N.A. become NaT."""
    value = str(value).strip()
    if value.lower() in {"present", "n.a.", "na", ""}:
        return pd.NaT
    value = re.sub(r",(?=\S)", ", ", value)          # 'Oct 15,2022' -> 'Oct 15, 2022'
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return pd.to_datetime(value, format=fmt)
        except ValueError:
            continue
    return pd.to_datetime(value, errors="coerce")


def parse_date_checked(value: str) -> pd.Timestamp:
    """parse_date plus a sanity check: dates with an impossible year become missing instead of being kept."""
    parsed = parse_date(value)
    if pd.notna(parsed) and not (VALID_YEARS[0] <= parsed.year <= VALID_YEARS[1]):
        return pd.NaT
    return parsed


def parse_rating(value) -> tuple[float, bool]:
    """'6.1(S)' -> (6.1, True) where True means the player came on as a substitute. 'N.A.' -> (nan, False)."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(\(S\))?\s*", str(value))
    if not match:
        return np.nan, False
    return float(match.group(1)), match.group(2) is not None


def categorise_injury(label: str) -> str:
    """Map one of ~100 free-text injury labels to a small set of categories."""
    text = str(label).strip().lower()
    for category, pattern in INJURY_RULES:
        if re.search(pattern, text):
            return category
    return "Other"


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardise types, text and missing values. One row per injury."""
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(lambda col: col.str.strip())
    df = df.replace({token: np.nan for token in NA_TOKENS})
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["FIFA rating"] = pd.to_numeric(df["FIFA rating"], errors="coerce")
    for col in [c for c in df.columns if c.endswith("_GD")]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Player ratings: split '6.1(S)' into a number and a substitute flag
    for col in [c for c in df.columns if c.endswith("Player_rating")]:
        parsed = df[col].map(parse_rating)
        df[col] = parsed.map(lambda t: t[0])
        df[col.replace("Player_rating", "is_sub")] = parsed.map(lambda t: t[1])
    df["injury_date"] = df["Date of Injury"].fillna("").map(parse_date_checked)
    df["return_date"] = df["Date of return"].fillna("").map(parse_date_checked)
    df["still_injured"] = df["Date of return"].fillna("").str.lower().eq("present")
    df["Name"] = df["Name"].str.replace(r"\s+", " ", regex=True).str.title()
    df["Position"] = df["Position"].str.strip()
    df["position_group"] = df["Position"].map(POSITION_GROUPS).fillna("Other")
    df["injury_type"] = df["Injury"].map(categorise_injury)
    df = df.rename(columns={"Team Name": "team", "Name": "player", "Position": "position", "Age": "age",
                            "Season": "season", "FIFA rating": "fifa_rating", "Injury": "injury_raw"})

    df = df.drop_duplicates().reset_index(drop=True)
    df.insert(0, "injury_id", df.index + 1)
    return df


def repair_return_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fix obvious year typos in the return date, e.g. 'Mar 7, 0202' (unreadable year) or a return date a year
    before the injury ('Jan 21, 2019' for an injury on 'Jan 1, 2020').
      1. Unreadable year  -> use the injury year (or the next year if that would still be before the injury).
      2. Return before injury -> add one year if that gives an absence of 0 to 365 days.
    Anything that cannot be repaired this way is left for the later checks. Returns (df, n_repaired)."""
    df = df.copy()
    repaired = 0
    for idx, row in df.iterrows():
        if row["still_injured"] or pd.isna(row["injury_date"]):
            continue
        ret, inj = row["return_date"], row["injury_date"]
        if pd.isna(ret):
            text = str(row["Date of return"])
            if text in ("nan", "None", "") or text.lower() == "present":
                continue
            parsed = pd.to_datetime(re.sub(r"\b\d{4}\b", str(inj.year), re.sub(r",(?=\S)", ", ", text)), errors="coerce")
            if pd.notna(parsed):
                if parsed < inj:
                    parsed = parsed + pd.DateOffset(years=1)
                df.at[idx, "return_date"] = parsed
                repaired += 1
        elif ret < inj:
            candidate = ret + pd.DateOffset(years=1)
            if 0 <= (candidate - inj).days <= 365:
                df.at[idx, "return_date"] = candidate
                repaired += 1
    return df, repaired


def resolve_result_gd_conflicts(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Logic check: a win needs GD > 0, a loss GD < 0 and a draw GD = 0.
    About 1% of matches break this rule. We cannot tell which field is wrong, so the goal difference is set
    to missing for those matches (the result, which is used for points, is kept). Returns (df, n_fixed)."""
    df = df.copy()
    fixed = 0
    for phase in PHASES:
        for n in MATCH_NUMBERS:
            res, gd = df[f"Match{n}_{phase}_Result"], df[f"Match{n}_{phase}_GD"]
            bad = ((res == "win") & (gd <= 0)) | ((res == "lose") & (gd >= 0)) | ((res == "draw") & gd.notna() & (gd != 0))
            df.loc[bad, f"Match{n}_{phase}_GD"] = np.nan
            fixed += int(bad.sum())
    return df, fixed


def _phase_mean(df: pd.DataFrame, phase: str, metric: str) -> pd.Series:
    """Row-wise mean over the three matches of a phase, ignoring missing matches."""
    cols = [f"Match{n}_{phase}_{metric}" for n in MATCH_NUMBERS if f"Match{n}_{phase}_{metric}" in df.columns]
    return df[cols].mean(axis=1, skipna=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the analytical columns used in the dashboard."""
    df = df.copy()

    # Duration. Return before injury is a data-entry error, so duration is set to NaN and flagged.
    df["days_out"] = (df["return_date"] - df["injury_date"]).dt.days
    df["date_issue"] = df["days_out"] < 0
    df.loc[df["date_issue"], "days_out"] = np.nan
    for phase in PHASES:
        for n in MATCH_NUMBERS:
            df[f"Match{n}_{phase}_points"] = df[f"Match{n}_{phase}_Result"].map(RESULT_POINTS)

    # Averages per phase (only over matches that exist - never filled with zero)
    for phase in PHASES:
        df[f"gd_{phase}"] = _phase_mean(df, phase, "GD")
        df[f"points_{phase}"] = _phase_mean(df, phase, "points")
    df["rating_before"] = _phase_mean(df, "before_injury", "Player_rating")
    df["rating_after"] = _phase_mean(df, "after_injury", "Player_rating")
    df["rating_change"] = df["rating_after"] - df["rating_before"]                     # positive = improved
    df["performance_drop_index"] = df["gd_before_injury"] - df["gd_missed_match"]       # positive = team worse without player
    df["points_drop"] = df["points_before_injury"] - df["points_missed_match"]

    # Severity bands from the football injury consensus statement (Fuller et al., 2006), based on days of absence
    df["severity"] = pd.cut(df["days_out"], bins=[-1, 1, 3, 7, 28, np.inf], labels=SEVERITY_LABELS)
    df["matches_missed"] = df[[f"Match{n}_missed_match_Result" for n in MATCH_NUMBERS]].notna().sum(axis=1)
    df["star_player"] = np.where(df["fifa_rating"] >= STAR_THRESHOLD, "Star (FIFA 82+)", "Squad player")
    df["age_group"] = pd.cut(df["age"], bins=[0, 22, 26, 30, 100], labels=["Under 23", "23-26", "27-30", "31+"])
    df["injury_month"] = df["injury_date"].dt.month_name().str[:3]
    df["injury_month_num"] = df["injury_date"].dt.month
    df["injury_year"] = df["injury_date"].dt.year
    return df


def to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape the 9 match slots per injury into rows: injury_id, phase, match_number, result, GD, rating..."""
    rows = []
    for phase in PHASES:
        for n in MATCH_NUMBERS:
            part = df[["injury_id", "player", "team", "season", "injury_type",
                       f"Match{n}_{phase}_Result", f"Match{n}_{phase}_Opposition", f"Match{n}_{phase}_GD"]].copy()
            part.columns = ["injury_id", "player", "team", "season", "injury_type", "result", "opposition", "gd"]
            rating_col = f"Match{n}_{phase}_Player_rating"
            part["rating"] = df[rating_col] if rating_col in df.columns else np.nan
            part["phase"] = phase
            part["match_number"] = n
            rows.append(part)
    long_df = pd.concat(rows, ignore_index=True).dropna(subset=["result"])
    long_df["points"] = long_df["result"].map(RESULT_POINTS)
    long_df["phase_label"] = long_df["phase"].map(PHASE_LABELS)
    # Chronological slot. In the source, "Match1" is the match NEAREST to the injury (before) or to the return (after),
    # e.g. Match1_before = last match played before the injury. So the timeline reads:
    #   before M3, M2, M1 -> absent M1, M2, M3 -> after M1, M2, M3
    order = {"before_injury": {3: 0, 2: 1, 1: 2}, "missed_match": {1: 3, 2: 4, 3: 5}, "after_injury": {1: 6, 2: 7, 3: 8}}
    long_df["slot"] = [order[p][n] for p, n in zip(long_df["phase"], long_df["match_number"])]
    long_df["slot_label"] = long_df["slot"].map(SLOT_LABELS)
    return long_df.sort_values(["injury_id", "slot"]).reset_index(drop=True)


def run_quality_checks(raw: pd.DataFrame, df: pd.DataFrame, long_df: pd.DataFrame, n_conflicts: int = 0, n_dates_repaired: int = 0) -> pd.DataFrame:
    """Return a small table of what was found and fixed. Useful evidence for the README and About tab."""
    sub_flags = int(sum(df[c].sum() for c in df.columns if c.endswith("is_sub")))
    na_cells = int(raw.apply(lambda col: col.str.strip().isin(NA_TOKENS)).sum().sum())
    checks = [
        ("Rows in raw file", len(raw), "-"),
        ("Rows after cleaning", len(df), "Exact duplicate rows removed" if len(raw) != len(df) else "No exact duplicates found"),
        ("Cells written as 'N.A.' in the raw file", na_cells, "Converted to real missing values"),
        ("Distinct player names (raw)", int(raw["Name"].str.strip().nunique()),
         f"{df['player'].nunique()} after fixing case and spacing (e.g. 'van' vs 'Van')"),
        ("Distinct injury labels (raw, ignoring case)", int(raw["Injury"].str.strip().str.lower().nunique()),
         f"Grouped into {df['injury_type'].nunique()} categories"),
        ("Players still injured ('Present')", int(df["still_injured"].sum()), "Flagged; excluded from duration statistics"),
        ("Return dates with a year typo", n_dates_repaired, "Repaired: impossible year ('Mar 7, 0202') or a return dated a year before the injury"),
        ("Return date still before injury date", int(df["date_issue"].sum()), "Duration set to missing, row kept"),
        ("Ratings marked as substitute '(S)'", sub_flags, "Split into rating + is_sub flag"),
        ("Matches where result and goal difference disagree", n_conflicts, "Goal difference set to missing, result kept"),
    ]
    return pd.DataFrame(checks, columns=["Check", "Count", "Action taken"])


def missing_value_report(raw: pd.DataFrame) -> pd.DataFrame:
    """Share of missing cells for each group of columns, and how each group is handled."""
    def share(cols):
        block = raw[cols].apply(lambda col: col.str.strip())
        return round(100 * block.isin(NA_TOKENS).to_numpy().mean(), 1), block.size
    groups = [
        ("Player rating before injury", [c for c in raw.columns if "before_injury" in c and c.endswith("Player_rating")],
         "Left missing. Average rating uses only the matches that have a rating (never filled with 0)."),
        ("Player rating after return", [c for c in raw.columns if "after_injury" in c and c.endswith("Player_rating")],
         "Left missing. Average rating uses only the matches that have a rating."),
        ("Goal difference (all 9 match slots)", [c for c in raw.columns if c.endswith("_GD")],
         "Left missing. Phase averages ignore missing matches; 59 conflicting values also set to missing."),
        ("Match result (all 9 match slots)", [c for c in raw.columns if c.endswith("_Result")],
         "Left missing. Points are only counted for matches that were played and recorded."),
        ("Return date", ["Date of return"],
         "'Present' = still injured (flagged); year typos repaired; the rest left missing."),
    ]
    rows = []
    for label, cols, how in groups:
        pct, cells = share(cols)
        rows.append((label, cells, pct, how))
    return pd.DataFrame(rows, columns=["Column group", "Cells", "Missing %", "How it was handled"])


def get_clean_data(path=None):
    """Run the full pipeline. Returns (clean_df, long_df, quality_table)."""
    raw = load_data(path)
    repaired, n_dates = repair_return_dates(clean_data(raw))
    cleaned, n_conflicts = resolve_result_gd_conflicts(repaired)
    df = engineer_features(cleaned)
    long_df = to_long_format(df)
    quality = run_quality_checks(raw, df, long_df, n_conflicts, n_dates)
    return df, long_df, quality


if __name__ == "__main__":
    d, l, q = get_clean_data()
    print(d.shape, l.shape)
    print(q.to_string(index=False))
