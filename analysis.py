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


CORR_LABELS = {"age": "Age", "fifa_rating": "FIFA rating", "days_out": "Days out", "rating_before": "Rating before",
               "rating_after": "Rating after", "rating_change": "Rating change", "performance_drop_index": "Team drop index",
               "points_drop": "Points drop"}


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of the numeric player and team metrics."""
    return df[list(CORR_LABELS)].corr(method="pearson").rename(index=CORR_LABELS, columns=CORR_LABELS).round(2)


def severity_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Team impact by injury severity band (football consensus bands based on days absent)."""
    out = (df.dropna(subset=["severity"])
             .groupby("severity", observed=True)
             .agg(cases=("injury_id", "count"), avg_days_out=("days_out", "mean"),
                  avg_drop_index=("performance_drop_index", "mean"), avg_points_drop=("points_drop", "mean"),
                  avg_rating_change=("rating_change", "mean"))
             .reset_index())
    return out.round(2)


def regression_model(df: pd.DataFrame) -> dict:
    """Standardised OLS regression of the team drop index on age, FIFA rating and log(days out).
    Coefficients are the change in the drop index (goals per match) for a 1 standard deviation change in the predictor."""
    data = df[["performance_drop_index", "age", "fifa_rating", "days_out"]].dropna()
    data = data.assign(log_days=np.log1p(data["days_out"]))
    names = {"age": "Age", "fifa_rating": "FIFA rating", "log_days": "Days out (log)"}
    n = len(data)
    X = data[list(names)]
    if n < 20 or (X.std(ddof=0) == 0).any():
        return {"n": n}
    Z = (X - X.mean()) / X.std(ddof=0)
    A = np.column_stack([np.ones(n), Z.values])
    y = data["performance_drop_index"].values
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - A @ beta
    k, dof = A.shape[1], n - A.shape[1]
    sigma2 = (resid @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(A.T @ A)))
    t_vals = beta / se
    p_vals = 2 * stats.t.sf(np.abs(t_vals), dof)
    r2 = 1 - (resid @ resid) / ((y - y.mean()) ** 2).sum()
    adj_r2 = 1 - (1 - r2) * (n - 1) / dof
    f_stat = (r2 / (k - 1)) / ((1 - r2) / dof)
    table = pd.DataFrame({"Predictor": list(names.values()), "Coefficient (goals per 1 SD)": beta[1:], "Std error": se[1:],
                          "t": t_vals[1:], "p-value": p_vals[1:]}).round(3)
    return {"n": n, "r2": r2, "adj_r2": adj_r2, "f": f_stat, "f_p": stats.f.sf(f_stat, k - 1, dof), "table": table}


def robustness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Each parametric test next to a non-parametric check that does not assume normal data.
    If both p-values lead to the same conclusion, the result does not depend on the normality assumption."""
    rows = []

    def add(question, param_name, p_param, nonparam_name, p_non, n):
        rows.append({"Question": question, "Parametric test": param_name, "p": p_param,
                     "Non-parametric check": nonparam_name, "p (check)": p_non, "n": n,
                     "Same conclusion at 5%?": "n/a" if np.isnan(p_param) or np.isnan(p_non) else ("Yes" if (p_param < 0.05) == (p_non < 0.05) else "No")})

    s = df["performance_drop_index"].dropna()
    if len(s) >= 10 and s.nunique() > 1:
        add("Team does worse without the player?", "One-sample t-test", stats.ttest_1samp(s, 0.0).pvalue,
            "Wilcoxon signed-rank", stats.wilcoxon(s, zero_method="wilcox").pvalue, len(s))
    both = df[["rating_before", "rating_after"]].dropna()
    if len(both) >= 10 and (both["rating_after"] - both["rating_before"]).nunique() > 1:
        add("Rating changes after the injury?", "Paired t-test", stats.ttest_rel(both["rating_after"], both["rating_before"]).pvalue,
            "Wilcoxon signed-rank", stats.wilcoxon(both["rating_after"], both["rating_before"]).pvalue, len(both))
    star = df.loc[df["star_player"].str.startswith("Star"), "performance_drop_index"].dropna()
    squad = df.loc[~df["star_player"].str.startswith("Star"), "performance_drop_index"].dropna()
    if len(star) >= 5 and len(squad) >= 5:
        add("Star players hurt the team more?", "Welch t-test", stats.ttest_ind(star, squad, equal_var=False).pvalue,
            "Mann-Whitney U", stats.mannwhitneyu(star, squad, alternative="two-sided").pvalue, len(star) + len(squad))
    ag = df[["age", "performance_drop_index"]].dropna()
    if len(ag) >= 10 and ag["age"].nunique() > 1 and ag["performance_drop_index"].nunique() > 1:
        add("Older players hurt the team more?", "Pearson correlation", stats.pearsonr(ag["age"], ag["performance_drop_index"])[1],
            "Spearman correlation", stats.spearmanr(ag["age"], ag["performance_drop_index"])[1], len(ag))
    out = pd.DataFrame(rows)
    if not out.empty:
        out[["p", "p (check)"]] = out[["p", "p (check)"]].round(3)
    return out


def correlation_strength(r: float) -> str:
    """Plain-English size of a correlation coefficient."""
    if r is None or np.isnan(r):
        return "not available"
    size = abs(r)
    word = "very weak" if size < 0.1 else "weak" if size < 0.3 else "moderate" if size < 0.5 else "strong"
    return f"{word} {'positive' if r > 0 else 'negative'}"


def significance_label(p: float) -> str:
    if np.isnan(p):
        return "not enough data"
    return "statistically significant (p < 0.05)" if p < 0.05 else "not statistically significant (p >= 0.05)"
