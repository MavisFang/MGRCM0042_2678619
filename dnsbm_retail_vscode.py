#!/usr/bin/env python3
"""
Retail Dynamic Network SBM (DNSBM)
----------------------------------
Implements the empirical model in the dissertation using Tone & Tsutsui (2014):

- 2 divisions, 5 periods (2020-2024)
- Division 1 input: Operating expense
- Division 1 -> Division 2 links: Employee, Store
  * baseline: as-input (LB)
- Division 2 output: Net Sales
- Division 2 carry-over: Inventories
  * baseline: free carry-over (CF), unscored, continuity only
- input orientation
- VRS in each division and period
- equal period weights and equal division weights
- backward lexicographic period-priority procedure (latest period first)

Sensitivity specifications included:
1. Alternative monetary conversion dataset
2. COGS as an additional Division 2 input
3. Stores fixed
4. Stores excluded
5. No inventory carry-over
6. Exclude Walmart
7. Exclude Costco, BJ's Wholesale Club, and Kesko

Requirements:
    pip install numpy pandas scipy

Example:
    python dnsbm_retail.py \
        --baseline main_cpi_fx.csv \
        --fx main_fx_cpi.csv \
        --cogs cogs_cpi_fx.csv \
        --out results
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import spearmanr


FRONTIER_TOL = 1e-7


@dataclass(frozen=True)
class ModelSpec:
    include_cogs: bool = False
    store_mode: str = "as_input"  # "as_input", "fixed", "excluded"
    carryover: bool = True
    lexicographic: bool = True


# -----------------------------------------------------------------------------
# Data I/O
# -----------------------------------------------------------------------------

def read_model_csv(path: str | Path) -> pd.DataFrame:
    """
    Read the user's CSV robustly.

    The supplied CSV exports contain 105 real observations followed by many
    comma-only rows. This reader stops at the first fully empty data row after
    observations start, avoiding the ~1,048,576-row spreadsheet tail.
    """
    path = Path(path)
    rows: List[dict] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")

        for row in reader:
            retail = (row.get("Retail") or "").strip()
            year = (row.get("year") or "").strip()

            if not retail and not year:
                if rows:
                    break
                continue

            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No observations found in {path}")

    required = [
        "Retail",
        "year",
        "Employee",
        "Store",
        "Operating expense",
        "Inventories",
        "Net Sales",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    numeric_cols = [c for c in df.columns if c != "Retail"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        )

    df = df.dropna(subset=["Retail", "year"]).copy()
    df["year"] = df["year"].astype(int)

    if df[required].isna().any().any():
        bad = df.loc[df[required].isna().any(axis=1), required]
        raise ValueError(f"Missing/non-numeric model data detected:\n{bad}")

    counts = df.groupby("Retail")["year"].nunique()
    if counts.nunique() != 1:
        raise ValueError("Retailers do not all have the same number of periods.")

    return df


def build_arrays(
    df: pd.DataFrame,
    retailers: Sequence[str],
    include_cogs: bool,
) -> Tuple[Dict[str, np.ndarray], List[int]]:
    years = sorted(df["year"].unique().tolist())
    year_index = {y: t for t, y in enumerate(years)}
    retail_index = {r: j for j, r in enumerate(retailers)}

    variables = [
        "Operating expense",
        "Employee",
        "Store",
        "Inventories",
        "Net Sales",
    ]
    if include_cogs:
        if "Cost of sales" not in df.columns:
            raise ValueError("include_cogs=True but 'Cost of sales' is absent.")
        variables.append("Cost of sales")

    arr = {
        var: np.zeros((len(years), len(retailers)), dtype=float)
        for var in variables
    }

    sub = df[df["Retail"].isin(retailers)].copy()
    for _, row in sub.iterrows():
        t = year_index[int(row["year"])]
        j = retail_index[row["Retail"]]
        for var in variables:
            arr[var][t, j] = float(row[var])

    for var, a in arr.items():
        if np.any(a <= 0):
            raise ValueError(
                f"All model values must be positive for SBM normalization. "
                f"Non-positive value found in {var}."
            )

    return arr, years


# -----------------------------------------------------------------------------
# LP construction and solution
# -----------------------------------------------------------------------------

def _normalize_rows(A: Sequence[np.ndarray], b: Sequence[float]):
    """Numerically scale constraints without changing the feasible set."""
    if len(A) == 0:
        return None, None

    A2 = np.asarray(A, dtype=float)
    b2 = np.asarray(b, dtype=float)
    scale = np.maximum(np.max(np.abs(A2), axis=1), np.abs(b2))
    scale[scale == 0] = 1.0
    return A2 / scale[:, None], b2 / scale


def solve_one_dmu(
    df: pd.DataFrame,
    target: str,
    spec: ModelSpec,
    retailers: Optional[Sequence[str]] = None,
) -> dict:
    """
    Solve the DNSBM LP for one target DMU.

    We use the input-oriented form of Tone & Tsutsui (2014). Because the
    observed values in the SBM denominators are constants, the objective is
    linear. Instead of explicitly creating scored slack variables, the code
    minimizes the equivalent projected-input ratios:

        D1 score(t) = projected OPEX / observed OPEX

        D2 score(t) = arithmetic mean of projected/observed scored inputs
                      (Employee, Store; plus COGS in the COGS sensitivity)

    Equal division weights imply:
        period score(t) = 0.5 * D1 score(t) + 0.5 * D2 score(t)

    Equal period weights imply:
        overall score = mean_t period score(t)
    """
    if retailers is None:
        retailers = df["Retail"].drop_duplicates().tolist()
    retailers = list(retailers)

    if target not in retailers:
        raise ValueError(f"Target {target!r} is not in the active sample.")
    if spec.store_mode not in {"as_input", "fixed", "excluded"}:
        raise ValueError("store_mode must be 'as_input', 'fixed', or 'excluded'.")

    arr, years = build_arrays(df, retailers, spec.include_cogs)
    n = len(retailers)
    T = len(years)
    o = retailers.index(target)

    # Decision variables are lambda_{j,k,t} only.
    # Flat layout: [all D1 lambdas] + [all D2 lambdas]
    nv = 2 * T * n

    def ix(k: int, t: int, j: int) -> int:
        return (k * T + t) * n + j

    # Objective coefficient vectors for overall, period, and division scores.
    c_overall = np.zeros(nv)
    c_period: List[np.ndarray] = []
    c_division = [np.zeros(nv), np.zeros(nv)]

    for t in range(T):
        pc = np.zeros(nv)

        # Division 1: one scored input, OPEX.
        for j in range(n):
            ratio = arr["Operating expense"][t, j] / arr["Operating expense"][t, o]
            pc[ix(0, t, j)] += 0.5 * ratio
            c_division[0][ix(0, t, j)] += (1.0 / T) * ratio

        # Division 2 scored inputs.
        d2_inputs = ["Employee"]
        if spec.store_mode == "as_input":
            d2_inputs.append("Store")
        if spec.include_cogs:
            d2_inputs.append("Cost of sales")

        m2_scored = len(d2_inputs)
        for var in d2_inputs:
            for j in range(n):
                ratio = arr[var][t, j] / arr[var][t, o]
                pc[ix(1, t, j)] += 0.5 * (1.0 / m2_scored) * ratio
                c_division[1][ix(1, t, j)] += (
                    (1.0 / T) * (1.0 / m2_scored) * ratio
                )

        c_period.append(pc)
        c_overall += (1.0 / T) * pc

    A_eq: List[np.ndarray] = []
    b_eq: List[float] = []
    A_ub: List[np.ndarray] = []
    b_ub: List[float] = []

    for t in range(T):
        # VRS: sum_j lambda_{j,k,t} = 1, separately for each division/period.
        for k in (0, 1):
            row = np.zeros(nv)
            for j in range(n):
                row[ix(k, t, j)] = 1.0
            A_eq.append(row)
            b_eq.append(1.0)

        # Division 1 input: projected OPEX <= observed OPEX.
        row = np.zeros(nv)
        for j in range(n):
            row[ix(0, t, j)] = arr["Operating expense"][t, j]
        A_ub.append(row)
        b_ub.append(arr["Operating expense"][t, o])

        # Employee as-input link.
        # Continuity: Z_employee * lambda_D1 = Z_employee * lambda_D2.
        row = np.zeros(nv)
        for j in range(n):
            row[ix(0, t, j)] = arr["Employee"][t, j]
            row[ix(1, t, j)] -= arr["Employee"][t, j]
        A_eq.append(row)
        b_eq.append(0.0)

        # As-input condition: projected employee <= observed employee.
        row = np.zeros(nv)
        for j in range(n):
            row[ix(1, t, j)] = arr["Employee"][t, j]
        A_ub.append(row)
        b_ub.append(arr["Employee"][t, o])

        # Store link treatment.
        if spec.store_mode == "as_input":
            # Continuity between D1 and D2.
            row = np.zeros(nv)
            for j in range(n):
                row[ix(0, t, j)] = arr["Store"][t, j]
                row[ix(1, t, j)] -= arr["Store"][t, j]
            A_eq.append(row)
            b_eq.append(0.0)

            # As-input condition: projected stores <= observed stores.
            row = np.zeros(nv)
            for j in range(n):
                row[ix(1, t, j)] = arr["Store"][t, j]
            A_ub.append(row)
            b_ub.append(arr["Store"][t, o])

        elif spec.store_mode == "fixed":
            # Fixed (non-discretionary) link: observed value on both sides.
            row = np.zeros(nv)
            for j in range(n):
                row[ix(0, t, j)] = arr["Store"][t, j]
            A_eq.append(row)
            b_eq.append(arr["Store"][t, o])

            row = np.zeros(nv)
            for j in range(n):
                row[ix(1, t, j)] = arr["Store"][t, j]
            A_eq.append(row)
            b_eq.append(arr["Store"][t, o])

        # COGS sensitivity: exogenous Division 2 input.
        if spec.include_cogs:
            row = np.zeros(nv)
            for j in range(n):
                row[ix(1, t, j)] = arr["Cost of sales"][t, j]
            A_ub.append(row)
            b_ub.append(arr["Cost of sales"][t, o])

        # Desirable output: projected Net Sales >= observed Net Sales.
        row = np.zeros(nv)
        for j in range(n):
            row[ix(1, t, j)] = -arr["Net Sales"][t, j]
        A_ub.append(row)
        b_ub.append(-arr["Net Sales"][t, o])

    # Free inventory carry-over in Division 2.
    # Tone & Tsutsui continuity for transition t -> t+1:
    # sum_j INV_{j,t} lambda_{j,2,t} = sum_j INV_{j,t} lambda_{j,2,t+1}
    # 2020 has no incoming 2019 initial-condition constraint.
    if spec.carryover:
        for t in range(T - 1):
            row = np.zeros(nv)
            for j in range(n):
                row[ix(1, t, j)] = arr["Inventories"][t, j]
                row[ix(1, t + 1, j)] -= arr["Inventories"][t, j]
            A_eq.append(row)
            b_eq.append(0.0)

    A_eq_n, b_eq_n = _normalize_rows(A_eq, b_eq)
    A_ub_n, b_ub_n = _normalize_rows(A_ub, b_ub)

    primary = linprog(
        c_overall,
        A_ub=A_ub_n,
        b_ub=b_ub_n,
        A_eq=A_eq_n,
        b_eq=b_eq_n,
        bounds=(0.0, None),
        method="highs",
    )
    if not primary.success:
        raise RuntimeError(f"LP failed for {target}: {primary.message}")

    theta = float(c_overall @ primary.x)
    x = primary.x.copy()

    # Tone & Tsutsui (2014) period-efficiency uniqueness procedure:
    # keep optimal overall efficiency fixed, then minimize period T, T-1, ..., 2.
    if spec.lexicographic:
        lex_Aeq = [row.copy() for row in A_eq_n]
        lex_beq = list(b_eq_n)
        lex_Aeq.append(c_overall.copy())
        lex_beq.append(theta)

        for t in range(T - 1, 0, -1):
            lex_res = linprog(
                c_period[t],
                A_ub=A_ub_n,
                b_ub=b_ub_n,
                A_eq=np.vstack(lex_Aeq),
                b_eq=np.asarray(lex_beq),
                bounds=(0.0, None),
                method="highs",
            )
            if not lex_res.success:
                raise RuntimeError(
                    f"Lexicographic LP failed for {target}, {years[t]}: "
                    f"{lex_res.message}"
                )

            period_value = float(c_period[t] @ lex_res.x)
            x = lex_res.x.copy()
            lex_Aeq.append(c_period[t].copy())
            lex_beq.append(period_value)

    period_scores = np.array([float(pc @ x) for pc in c_period])
    division_scores = np.array([float(dc @ x) for dc in c_division])

    # Numerical identities implied by equal weights/input orientation.
    if not np.isclose(theta, period_scores.mean(), atol=1e-7):
        raise RuntimeError("Overall score != arithmetic mean of period scores.")
    if not np.isclose(theta, division_scores.mean(), atol=1e-7):
        raise RuntimeError("Overall score != arithmetic mean of divisional scores.")

    return {
        "target": target,
        "overall": theta,
        "period": period_scores,
        "division": division_scores,
        "x": x,
        "arrays": arr,
        "years": years,
        "retailers": retailers,
    }


def run_model(
    df: pd.DataFrame,
    spec: ModelSpec,
    retailers: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    if retailers is None:
        retailers = df["Retail"].drop_duplicates().tolist()
    retailers = list(retailers)

    rows = []
    solutions: Dict[str, dict] = {}

    for retailer in retailers:
        sol = solve_one_dmu(df, retailer, spec, retailers)
        solutions[retailer] = sol

        row = {
            "Retailer": retailer,
            "Overall": sol["overall"],
            "Division 1": sol["division"][0],
            "Division 2": sol["division"][1],
        }
        for year, score in zip(sol["years"], sol["period"]):
            row[str(year)] = score
        rows.append(row)

    out = pd.DataFrame(rows)

    # Competition ranking used in the dissertation table: all frontier DMUs rank 1,
    # and the next DMU starts after the number of tied frontier units.
    score_for_rank = out["Overall"].copy()
    score_for_rank.loc[score_for_rank >= 1.0 - FRONTIER_TOL] = 1.0
    out["Rank"] = score_for_rank.rank(ascending=False, method="min").astype(int)

    order = ["Retailer", "Overall", "Rank"] + [
        str(y) for y in sorted(df["year"].unique())
    ] + ["Division 1", "Division 2"]
    return out[order], solutions


# -----------------------------------------------------------------------------
# Diagnostics and reporting
# -----------------------------------------------------------------------------

def selected_projection_slacks(solution: dict) -> pd.DataFrame:
    """
    Raw/projected values for the selected optimal solution.

    For the input-oriented model, OPEX, Employee, and Store relative slacks are
    scored input excesses. Inventory deviation and Sales expansion are not scored
    in the input-oriented objective and should not be interpreted as unique targets.
    """
    target = solution["target"]
    retailers = solution["retailers"]
    years = solution["years"]
    arr = solution["arrays"]
    x = solution["x"]

    n = len(retailers)
    T = len(years)
    o = retailers.index(target)

    def ix(k: int, t: int, j: int) -> int:
        return (k * T + t) * n + j

    rows = []
    for t, year in enumerate(years):
        l1 = np.array([x[ix(0, t, j)] for j in range(n)])
        l2 = np.array([x[ix(1, t, j)] for j in range(n)])

        observed_opex = arr["Operating expense"][t, o]
        observed_emp = arr["Employee"][t, o]
        observed_store = arr["Store"][t, o]
        observed_sales = arr["Net Sales"][t, o]
        observed_inv = arr["Inventories"][t, o]

        proj_opex = float(arr["Operating expense"][t] @ l1)
        proj_emp = float(arr["Employee"][t] @ l2)
        proj_store = float(arr["Store"][t] @ l2)
        proj_sales = float(arr["Net Sales"][t] @ l2)
        proj_inv = float(arr["Inventories"][t] @ l2)

        rows.append(
            {
                "Retailer": target,
                "Year": year,
                "Observed OPEX": observed_opex,
                "Projected OPEX": proj_opex,
                "OPEX relative slack": (observed_opex - proj_opex) / observed_opex,
                "Observed Employee": observed_emp,
                "Projected Employee": proj_emp,
                "Employee relative slack": (observed_emp - proj_emp) / observed_emp,
                "Observed Store": observed_store,
                "Projected Store": proj_store,
                "Store relative slack": (observed_store - proj_store) / observed_store,
                "Observed Sales": observed_sales,
                "Selected projected Sales": proj_sales,
                "Selected Sales expansion": (proj_sales - observed_sales) / observed_sales,
                "Observed Inventory": observed_inv,
                "Selected projected Inventory": proj_inv,
                "Selected Inventory deviation": (observed_inv - proj_inv) / observed_inv,
            }
        )

    return pd.DataFrame(rows)


def clamp_frontier(scores: pd.Series, tol: float = 1e-6) -> np.ndarray:
    a = scores.to_numpy(dtype=float).copy()
    a[a >= 1.0 - tol] = 1.0
    return a


def spearman_against_baseline(
    baseline: pd.DataFrame,
    alternative: pd.DataFrame,
) -> float:
    """
    Standard Spearman rho using full-precision scores.
    Solver values numerically equal to one are clamped to exactly one so frontier
    DMUs are treated as ties, as stated in the dissertation.
    """
    common = [r for r in alternative["Retailer"] if r in set(baseline["Retailer"])]
    b = baseline.set_index("Retailer").loc[common, "Overall"]
    a = alternative.set_index("Retailer").loc[common, "Overall"]
    return float(spearmanr(clamp_frontier(b), clamp_frontier(a)).statistic)


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in ["Operating expense", "Inventories", "Net Sales", "Employee", "Store"]:
        x = df[var].astype(float)
        unit = "physical"
        if var in {"Operating expense", "Inventories", "Net Sales"}:
            x = x / 1e6
            unit = "USD million"
        rows.append(
            {
                "Variable": var,
                "Unit": unit,
                "Mean": x.mean(),
                "Std. dev.": x.std(ddof=1),
                "Min": x.min(),
                "Max": x.max(),
            }
        )
    return pd.DataFrame(rows)


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Operating expense", "Employee", "Store", "Inventories", "Net Sales"]
    return df[cols].corr(method="pearson")


def sample_structure_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["Net sales / staff (US$'000)"] = x["Net Sales"] / x["Employee"] / 1000.0
    x["Net sales / store (US$m)"] = x["Net Sales"] / x["Store"] / 1e6
    x["OPEX / net sales"] = x["Operating expense"] / x["Net Sales"]

    groups = [
        ("Warehouse club", "Costco", x[x["Retail"] == "Costco"]),
        ("Warehouse club", "BJ’s Wholesale Club", x[x["Retail"] == "BJ’s Wholesale Club"]),
        ("Franchise", "Kesko", x[x["Retail"] == "Kesko"]),
        (
            "Comparison group",
            "Remaining 18 retailers",
            x[~x["Retail"].isin(["Costco", "BJ’s Wholesale Club", "Kesko"])],
        ),
    ]

    rows = []
    for business_model, retailer, g in groups:
        rows.append(
            {
                "Business model": business_model,
                "Retailer": retailer,
                "Net sales / staff (US$'000)": g["Net sales / staff (US$'000)"].mean(),
                "Net sales / store (US$m)": g["Net sales / store (US$m)"].mean(),
                "OPEX / net sales": g["OPEX / net sales"].mean(),
            }
        )
    return pd.DataFrame(rows)


def summarize_spec(
    name: str,
    result: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict:
    return {
        "Specification": name,
        "Mean": result["Overall"].mean(),
        "Mean D1": result["Division 1"].mean(),
        "Mean D2": result["Division 2"].mean(),
        "Frontier": int((result["Overall"] >= 1.0 - FRONTIER_TOL).sum()),
        "Spearman": spearman_against_baseline(baseline, result),
    }


# -----------------------------------------------------------------------------
# Main reproducibility run
# -----------------------------------------------------------------------------

def main() -> None:
    """
    VS Code friendly entry point.

    Put this .py file and the three CSV files in the SAME folder, then click
    the VS Code "Run Python File" button. No command-line arguments are needed.
    """
    base_dir = Path(__file__).resolve().parent

    # ------------------------------------------------------------------
    # FILE NAMES: change only these three lines if your CSV names differ.
    # ------------------------------------------------------------------
    baseline_path = base_dir / "main_cpi_fx.csv"
    fx_path = base_dir / "main_fx_cpi.csv"
    cogs_path = base_dir / "cogs_cpi_fx.csv"

    out_dir = base_dir / "dnsbm_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_path.exists():
        raise FileNotFoundError(
            "Baseline CSV was not found.\n"
            f"Expected: {baseline_path}\n\n"
            "Put main_cpi_fx.csv in the same folder as this Python file, "
            "or edit baseline_path near the bottom of the script."
        )

    print(f"Python script folder: {base_dir}")
    print(f"Baseline data: {baseline_path.name}")

    baseline_df = read_model_csv(baseline_path)
    base_spec = ModelSpec()
    baseline, base_solutions = run_model(baseline_df, base_spec)

    # Core outputs.
    baseline.to_csv(out_dir / "baseline_efficiency.csv", index=False)
    descriptive_statistics(baseline_df).to_csv(
        out_dir / "descriptive_statistics.csv", index=False
    )
    correlation_table(baseline_df).to_csv(out_dir / "correlations.csv")
    sample_structure_indicators(baseline_df).to_csv(
        out_dir / "sample_structure_indicators.csv", index=False
    )

    # Selected-projection slacks for all retailers.
    all_slacks = pd.concat(
        [selected_projection_slacks(base_solutions[r]) for r in baseline["Retailer"]],
        ignore_index=True,
    )
    all_slacks.to_csv(out_dir / "baseline_selected_projection_slacks.csv", index=False)

    sensitivity_rows = []

    # Alternative FX/CPI dataset is optional.
    if fx_path.exists():
        fx_df = read_model_csv(fx_path)
        fx_res, _ = run_model(fx_df, base_spec)
        fx_res.to_csv(out_dir / "sensitivity_fx_then_cpi.csv", index=False)
        sensitivity_rows.append(
            summarize_spec("Annual FX then US CPI", fx_res, baseline)
        )
    else:
        print(f"WARNING: {fx_path.name} not found; FX/CPI sensitivity skipped.")

    # COGS dataset is optional.
    if cogs_path.exists():
        cogs_df = read_model_csv(cogs_path)
        cogs_res, _ = run_model(cogs_df, ModelSpec(include_cogs=True))
        cogs_res.to_csv(out_dir / "sensitivity_cogs.csv", index=False)
        sensitivity_rows.append(
            summarize_spec("Division 2 input: COGS", cogs_res, baseline)
        )
    else:
        print(f"WARNING: {cogs_path.name} not found; COGS sensitivity skipped.")

    fixed_res, _ = run_model(baseline_df, ModelSpec(store_mode="fixed"))
    fixed_res.to_csv(out_dir / "sensitivity_stores_fixed.csv", index=False)
    sensitivity_rows.append(summarize_spec("Stores fixed", fixed_res, baseline))

    no_store_res, _ = run_model(baseline_df, ModelSpec(store_mode="excluded"))
    no_store_res.to_csv(out_dir / "sensitivity_stores_excluded.csv", index=False)
    sensitivity_rows.append(
        summarize_spec("Stores excluded", no_store_res, baseline)
    )

    no_carry_res, _ = run_model(baseline_df, ModelSpec(carryover=False))
    no_carry_res.to_csv(out_dir / "sensitivity_no_carryover.csv", index=False)
    sensitivity_rows.append(
        summarize_spec("No inventory carry-over", no_carry_res, baseline)
    )

    all_retailers = baseline_df["Retail"].drop_duplicates().tolist()

    no_walmart = [r for r in all_retailers if r != "Walmart"]
    no_walmart_res, _ = run_model(baseline_df, base_spec, no_walmart)
    no_walmart_res.to_csv(out_dir / "sensitivity_exclude_walmart.csv", index=False)
    sensitivity_rows.append(
        summarize_spec("Exclude Walmart", no_walmart_res, baseline)
    )

    no_three = [
        r
        for r in all_retailers
        if r not in {"Costco", "BJ’s Wholesale Club", "Kesko"}
    ]
    no_three_res, _ = run_model(baseline_df, base_spec, no_three)
    no_three_res.to_csv(
        out_dir / "sensitivity_exclude_costco_bj_kesko.csv", index=False
    )
    sensitivity_rows.append(
        summarize_spec("Exclude Costco/BJ/Kesko", no_three_res, baseline)
    )

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(out_dir / "sensitivity_summary.csv", index=False)

    # Table 10-style common-DMU baseline comparisons.
    base_index = baseline.set_index("Retailer")
    table10_rows = []
    for label, res in [
        ("Exclude Walmart", no_walmart_res),
        ("Exclude Costco/BJ/Kesko", no_three_res),
    ]:
        common = res["Retailer"].tolist()
        common_base = base_index.loc[common].reset_index()
        table10_rows.append(
            {
                "Specification": label,
                "Basis": "Common DMU baseline",
                "DMU": len(common),
                "Mean": common_base["Overall"].mean(),
                "Mean D1": common_base["Division 1"].mean(),
                "Mean D2": common_base["Division 2"].mean(),
                "Frontier": int(
                    (common_base["Overall"] >= 1.0 - FRONTIER_TOL).sum()
                ),
            }
        )
        table10_rows.append(
            {
                "Specification": label,
                "Basis": "Re-estimated after exclusion",
                "DMU": len(common),
                "Mean": res["Overall"].mean(),
                "Mean D1": res["Division 1"].mean(),
                "Mean D2": res["Division 2"].mean(),
                "Frontier": int((res["Overall"] >= 1.0 - FRONTIER_TOL).sum()),
            }
        )
    pd.DataFrame(table10_rows).to_csv(
        out_dir / "sample_exclusion_table10.csv", index=False
    )

    # Console summary.
    year_cols = [str(y) for y in sorted(baseline_df["year"].unique())]
    print("\nBASELINE")
    print(
        baseline.sort_values(["Overall", "Retailer"], ascending=[False, True])[
            ["Retailer", "Overall", "Rank"]
            + year_cols
            + ["Division 1", "Division 2"]
        ].to_string(index=False, float_format=lambda z: f"{z:.6f}")
    )

    print("\nBaseline means:")
    print(f"Overall = {baseline['Overall'].mean():.6f}")
    print(f"D1      = {baseline['Division 1'].mean():.6f}")
    print(f"D2      = {baseline['Division 2'].mean():.6f}")
    for y in year_cols:
        print(
            f"{y} mean = {baseline[y].mean():.6f}; "
            f"median = {baseline[y].median():.6f}"
        )

    if not sensitivity.empty:
        print("\nSENSITIVITY SUMMARY")
        print(sensitivity.to_string(index=False, float_format=lambda z: f"{z:.6f}"))

    print("\nDONE")
    print(f"Results folder: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
