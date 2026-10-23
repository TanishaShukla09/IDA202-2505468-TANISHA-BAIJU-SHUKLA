"""
analysis.py
-----------
Aggregations and statistical tests used by the dashboard and the EDA notebook.
All functions take the cleaned DataFrame from data_processing.get_clean_data() and return small tables
or dictionaries, so they are easy to test and to reuse in the notebook.

Sign convention used everywhere
    performance_drop_index (PDI) = average goal difference BEFORE the injury
                                   minus average goal difference while the player was ABSENT.
    PDI > 0  -> the team did worse without the player.   PDI < 0 -> the team did better without him.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, median, standard deviation, min, max and sample size for the headline metrics."""
    metrics = {
        "days_out": "Days out",
        "performance_drop_index": "Team performance drop index",
        "points_drop": "Points drop (per match)",
        "rating_before": "Player rating before",
        "rating_after": "Player rating after",
        "rating_change": "Rating change (after - before)",
    }
    rows = []
    for col, label in metrics.items():
        s = df[col].dropna()
        rows.append({"Metric": label, "n": len(s), "Mean": s.mean(), "Median": s.median(),
                     "Std dev": s.std(), "Min": s.min(), "Max": s.max()})
    return pd.DataFrame(rows).round(2)


def injury_type_impact(df: pd.DataFrame, min_cases: int = 5) -> pd.DataFrame:
    """Average impact by injury category. Categories with fewer than `min_cases` cases are dropped
    because averages over a handful of injuries are mostly noise."""
    out = (df.groupby("injury_type")
             .agg(cases=("injury_id", "count"),
                  avg_drop_index=("performance_drop_index", "mean"),
                  avg_points_drop=("points_drop", "mean"),
                  avg_days_out=("days_out", "mean"),
                  avg_rating_change=("rating_change", "mean"))
             .query("cases >= @min_cases")
             .sort_values("avg_drop_index", ascending=False)
             .reset_index())
    return out.round(2)


def club_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Injuries and impact per club."""
    out = (df.groupby("team")
             .agg(injuries=("injury_id", "count"),
                  players_hit=("player", "nunique"),
                  avg_days_out=("days_out", "mean"),
                  avg_drop_index=("performance_drop_index", "mean"),
                  avg_points_drop=("points_drop", "mean"))
             .sort_values("injuries", ascending=False)
             .reset_index())
    return out.round(2)


def absence_record(long_df: pd.DataFrame) -> pd.DataFrame:
    """Share of wins / draws / losses per club in matches the injured player missed."""
    missed = long_df[long_df["phase"] == "missed_match"]
    counts = missed.groupby(["team", "result"]).size().unstack(fill_value=0)
    for col in ["win", "draw", "lose"]:
        if col not in counts:
            counts[col] = 0
    counts["matches"] = counts[["win", "draw", "lose"]].sum(axis=1)
    pct = counts[["win", "draw", "lose"]].div(counts["matches"], axis=0) * 100
    pct.columns = ["Win %", "Draw %", "Loss %"]
    pct["Matches"] = counts["matches"]
    return pct.round(1).reset_index().rename(columns={"team": "Club"})


def phase_performance(long_df: pd.DataFrame) -> pd.DataFrame:
    """Points per match and goal difference per match in each phase (before / absent / after)."""
    out = (long_df.groupby("phase")
                  .agg(matches=("points", "count"), points_per_match=("points", "mean"), avg_gd=("gd", "mean"))
                  .reindex(["before_injury", "missed_match", "after_injury"]))
    out.index = ["Before injury", "Player absent", "After return"]
    return out.round(2).reset_index().rename(columns={"index": "Phase"})


def player_phase_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Group by player: injuries, days out and average rating / goal difference in each injury phase
    (before, during absence, after return) - the 'group by Player Name' step in the brief."""
    out = (df.groupby(["player", "team"])
             .agg(injuries=("injury_id", "count"),
                  total_days_out=("days_out", "sum"),
                  rating_before=("rating_before", "mean"),
                  rating_after=("rating_after", "mean"),
                  gd_before=("gd_before_injury", "mean"),
                  gd_during_absence=("gd_missed_match", "mean"),
                  gd_after=("gd_after_injury", "mean"))
             .reset_index())
    out["rating_change"] = out["rating_after"] - out["rating_before"]
    return out.sort_values("injuries", ascending=False).round(2)


def most_injured_players(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    return (df.groupby(["player", "team"])
              .agg(injuries=("injury_id", "count"), total_days_out=("days_out", "sum"))
              .sort_values(["injuries", "total_days_out"], ascending=False)
              .head(top).reset_index())


def comeback_leaderboard(df: pd.DataFrame, top: int = 10, min_cases: int = 1, worst: bool = False) -> pd.DataFrame:
    """Players ranked by average change in match rating after returning (after - before).
    Needs a valid rating both before and after the injury."""
    valid = df.dropna(subset=["rating_change"])
    out = (valid.groupby(["player", "team"])
                .agg(injuries=("injury_id", "count"),
                     rating_before=("rating_before", "mean"),
                     rating_after=("rating_after", "mean"),
                     rating_change=("rating_change", "mean"))
                .query("injuries >= @min_cases")
                .sort_values("rating_change", ascending=worst)
                .head(top).reset_index())
    return out.round(2)


def pre_post_pivot(df: pd.DataFrame, index: str = "team") -> pd.DataFrame:
    """pivot_table comparing average player rating before and after injury."""
    pivot = pd.pivot_table(df, index=index, values=["rating_before", "rating_after", "rating_change"],
                           aggfunc="mean")
    return pivot[["rating_before", "rating_after", "rating_change"]].round(2).reset_index()


def month_club_heatmap(df: pd.DataFrame, month_order: list[str]) -> pd.DataFrame:
    """Injury counts: clubs as rows, calendar months as columns."""
    table = pd.crosstab(df["team"], df["injury_month"]).reindex(columns=month_order, fill_value=0)
    return table.loc[table.sum(axis=1).sort_values(ascending=False).index]


def top_impact_injuries(df: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    """The individual injuries after which the team did worst without the player."""
    cols = ["player", "team", "season", "injury_raw", "days_out", "performance_drop_index", "points_drop"]
    return df.dropna(subset=["performance_drop_index"]).nlargest(top, "performance_drop_index")[cols].round(2)


def age_correlation(df: pd.DataFrame) -> dict:
    """Pearson correlation between age and PDI, with a least-squares trend line."""
    data = df[["age", "performance_drop_index"]].dropna()
    if len(data) < 3 or data["age"].nunique() < 2 or data["performance_drop_index"].nunique() < 2:
        return {"n": len(data), "r": np.nan, "p": np.nan, "slope": np.nan, "intercept": np.nan}
    r, p = stats.pearsonr(data["age"], data["performance_drop_index"])
    slope, intercept = np.polyfit(data["age"], data["performance_drop_index"], 1)
    return {"n": len(data), "r": r, "p": p, "slope": slope, "intercept": intercept}


def paired_rating_test(df: pd.DataFrame) -> dict:
    """Paired t-test: is the player's rating after returning different from before the injury?"""
    data = df[["rating_before", "rating_after"]].dropna()
    if len(data) < 3:
        return {"n": len(data)}
    t, p = stats.ttest_rel(data["rating_after"], data["rating_before"])
    diff = data["rating_after"] - data["rating_before"]
    ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff))
    return {"n": len(data), "mean_diff": diff.mean(), "t": t, "p": p, "ci_low": ci[0], "ci_high": ci[1],
            "cohens_dz": diff.mean() / diff.std(ddof=1)}


def team_drop_test(df: pd.DataFrame) -> dict:
    """One-sample t-test: is the average PDI different from 0 (i.e. does the team really do worse without the player)?"""
    s = df["performance_drop_index"].dropna()
    if len(s) < 3:
        return {"n": len(s)}
    t, p = stats.ttest_1samp(s, 0.0)
    ci = stats.t.interval(0.95, len(s) - 1, loc=s.mean(), scale=stats.sem(s))
    return {"n": len(s), "mean": s.mean(), "t": t, "p": p, "ci_low": ci[0], "ci_high": ci[1]}


def star_vs_squad_test(df: pd.DataFrame) -> dict:
    """Welch t-test comparing PDI for star players with other squad players."""
    star = df.loc[df["star_player"].str.startswith("Star"), "performance_drop_index"].dropna()
    squad = df.loc[~df["star_player"].str.startswith("Star"), "performance_drop_index"].dropna()
    if len(star) < 3 or len(squad) < 3:
        return {"n_star": len(star), "n_squad": len(squad)}
    t, p = stats.ttest_ind(star, squad, equal_var=False)
    return {"n_star": len(star), "n_squad": len(squad), "mean_star": star.mean(), "mean_squad": squad.mean(),
            "t": t, "p": p}


def significance_label(p: float) -> str:
    if np.isnan(p):
        return "not enough data"
    return "statistically significant (p < 0.05)" if p < 0.05 else "not statistically significant (p >= 0.05)"
