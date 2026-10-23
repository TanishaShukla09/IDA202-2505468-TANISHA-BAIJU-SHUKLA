from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).parent
DATA_PATH = next((p for p in [_HERE / "data" / "player_injuries_impact.csv", _HERE / "player_injuries_impact.csv"]
                  if p.exists()), _HERE / "data" / "player_injuries_impact.csv")

NA_TOKENS = ["N.A.", "N.A", "n.a.", "NA", "N/A", ""]   # text used for "missing" in the raw file
RESULT_POINTS = {"win": 3, "draw": 1, "lose": 0}        # league points for a result
PHASES = ["before_injury", "missed_match", "after_injury"]
PHASE_LABELS = {"before_injury": "Before injury", "missed_match": "Player absent", "after_injury": "After return"}
MATCH_NUMBERS = [1, 2, 3]
SLOT_LABELS = {0: "Before -3", 1: "Before -2", 2: "Before -1", 3: "Absent 1", 4: "Absent 2",
               5: "Absent 3", 6: "After +1", 7: "After +2", 8: "After +3"}
STAR_THRESHOLD = 82            # FIFA rating at or above this = "star player" (roughly the top fifth)
VALID_RESULTS = {"win", "draw", "lose"}


def required_columns() -> set[str]:
    """Columns needed by the cleaning and feature-engineering pipeline."""
    required = {"Team Name", "Name", "Position", "Age", "Season", "FIFA rating",
                "Injury", "Date of Injury", "Date of return"}
    for phase in PHASES:
        for n in MATCH_NUMBERS:
            # Results and goal difference are required for team-performance analysis.
            # Player ratings are optional because some dataset versions omit ratings
            # for matches missed during an injury; downstream calculations can handle NaN.
            required.update({f"Match{n}_{phase}_Result", f"Match{n}_{phase}_GD"})
    return required

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
    `path` can be a file path or an open file object; None means the default dataset in data/."""
    if path is None:
        path = DATA_PATH
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
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("clean_data() expects a pandas DataFrame.")
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        duplicates = sorted(set(df.columns[df.columns.duplicated()].tolist()))
        raise ValueError(f"The source file has duplicate column names: {duplicates}")
    missing = sorted(required_columns() - set(df.columns))
    if missing:
        preview = ", ".join(missing[:8]) + (" …" if len(missing) > 8 else "")
        raise ValueError(
            "The injury dataset is missing required columns: " + preview + ". "
            "Check that you selected the expected player_injuries_impact.csv file."
        )
    # Some dataset versions omit ratings for matches missed during an injury.
    # Add absent rating columns as missing values so other analyses still work.
    # Rating-based summaries will remain missing rather than being fabricated.
    for phase in PHASES:
        for n in MATCH_NUMBERS:
            rating_col = f"Match{n}_{phase}_Player_rating"
            if rating_col not in df.columns:
                df[rating_col] = np.nan
    # Strip text columns only. This keeps clean_data safe when called directly
    # with a DataFrame that already contains numeric columns (not just load_data's strings).
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(lambda value: value.strip() if isinstance(value, str) else value)
    # Treat common missing-value spellings consistently, regardless of case.
    missing_tokens = {token.lower() for token in NA_TOKENS}
    df = df.applymap(lambda value: np.nan if isinstance(value, str) and value.lower() in missing_tokens else value)
    # Parse numeric columns explicitly. Values outside plausible domain ranges are
    # treated as missing rather than allowed to distort averages and charts.
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df.loc[~df["Age"].between(15, 60), "Age"] = np.nan
    df["FIFA rating"] = pd.to_numeric(df["FIFA rating"], errors="coerce")
    df.loc[~df["FIFA rating"].between(0, 100), "FIFA rating"] = np.nan
    for col in [c for c in df.columns if c.endswith("_GD")]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Match ratings are parsed below; malformed values become NaN.

    # Player ratings: split '6.1(S)' into a number and a substitute flag
    for col in [c for c in df.columns if c.endswith("Player_rating")]:
        parsed = df[col].map(parse_rating)
        values = parsed.map(lambda t: t[0])
        # Football match ratings are conventionally on a 0–10 scale. Invalid
        # out-of-range values are excluded from rating summaries.
        values = values.where(values.between(0, 10))
        df[col] = values
        df[col.replace("Player_rating", "is_sub")] = parsed.map(lambda t: t[1]) & values.notna()
    df["injury_date"] = df["Date of Injury"].fillna("").map(parse_date)
    df["return_date"] = df["Date of return"].fillna("").map(parse_date)
    df["still_injured"] = df["Date of return"].fillna("").astype(str).str.lower().eq("present")
    # Result labels are categorical inputs to points calculations; normalise their case/spacing.
    result_cols = [c for c in df.columns if c.endswith("_Result")]
    for col in result_cols:
        df[col] = df[col].astype("string").str.strip().str.lower().replace({"": pd.NA})
        # Unknown result labels must not silently enter points-per-match metrics.
        df.loc[~df[col].isin(VALID_RESULTS) & df[col].notna(), col] = pd.NA
    df["Name"] = df["Name"].astype("string").str.strip().str.title()
    df["Position"] = df["Position"].astype("string").str.strip()
    df["Team Name"] = df["Team Name"].astype("string").str.strip()
    df["Season"] = df["Season"].astype("string").str.strip()
    df["Injury"] = df["Injury"].astype("string").str.strip()
    df["position_group"] = df["Position"].map(POSITION_GROUPS).fillna("Other")
    df["injury_type"] = df["Injury"].map(categorise_injury)
    df = df.rename(columns={"Team Name": "team", "Name": "player", "Position": "position", "Age": "age",
                            "Season": "season", "FIFA rating": "fifa_rating", "Injury": "injury_raw"})

    df = df.drop_duplicates().reset_index(drop=True)
    df.insert(0, "injury_id", df.index + 1)
    return df


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


def run_quality_checks(raw: pd.DataFrame, df: pd.DataFrame, long_df: pd.DataFrame, n_conflicts: int = 0) -> pd.DataFrame:
    """Return a small table of what was found and fixed. Useful evidence for the README and About tab."""
    sub_flags = int(sum(df[c].sum() for c in df.columns if c.endswith("is_sub")))
    na_cells = int(raw.apply(lambda col: col.astype("string").str.strip().str.lower().isin(
        {token.lower() for token in NA_TOKENS})).sum().sum())
    invalid_age = int(pd.to_numeric(raw["Age"], errors="coerce").notna().sum() - df["age"].notna().sum())
    invalid_fifa = int(pd.to_numeric(raw["FIFA rating"], errors="coerce").notna().sum() - df["fifa_rating"].notna().sum())
    invalid_results = 0
    for col in [c for c in raw.columns if c.endswith("_Result")]:
        values = raw[col].astype("string").str.strip().str.lower()
        values = values.mask(values.isin({token.lower() for token in NA_TOKENS}))
        invalid_results += int((~values.isin(VALID_RESULTS) & values.notna()).sum())
    checks = [
        ("Rows in raw file", len(raw), "-"),
        ("Rows after cleaning", len(df), "Exact duplicate rows removed" if len(raw) != len(df) else "No exact duplicates found"),
        ("Cells written as 'N.A.' in the raw file", na_cells, "Converted to real missing values"),
        ("Age values invalid or outside 15–60", invalid_age, "Set to missing; retained the injury record"),
        ("FIFA ratings invalid or outside 0–100", invalid_fifa, "Set to missing; retained the injury record"),
        ("Unknown match-result labels", invalid_results, "Excluded from result-based metrics"),
        ("Distinct injury labels (raw, ignoring case)", int(raw["Injury"].str.strip().str.lower().nunique()),
         f"Grouped into {df['injury_type'].nunique()} categories"),
        ("Injury dates that could not be parsed", int(df["injury_date"].isna().sum()), "Flexible date parser handles mixed formats"),
        ("Players still injured ('Present')", int(df["still_injured"].sum()), "Flagged; excluded from duration statistics"),
        ("Return date before injury date", int(df["date_issue"].sum()), "Duration set to missing, row kept"),
        ("Ratings marked as substitute '(S)'", sub_flags, "Split into rating + is_sub flag"),
        ("Matches where result and goal difference disagree", n_conflicts, "Goal difference set to missing, result kept"),
    ]
    return pd.DataFrame(checks, columns=["Check", "Count", "Action taken"])


def get_clean_data(path=None):
    """Run the full pipeline. Returns (clean_df, long_df, quality_table).

    `path` may be a filesystem path or a file-like object (e.g. Streamlit upload).
    The return signature is intentionally stable for compatibility with app.py.
    """
    raw = load_data(path)
    if raw.empty:
        raise ValueError("The selected CSV contains no data rows.")
    cleaned, n_conflicts = resolve_result_gd_conflicts(clean_data(raw))
    df = engineer_features(cleaned)
    long_df = to_long_format(df)
    quality = run_quality_checks(raw, df, long_df, n_conflicts)
    return df, long_df, quality


if __name__ == "__main__":
    d, l, q = get_clean_data()
    print(d.shape, l.shape)
    print(q.to_string(index=False))