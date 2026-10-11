"""
Loan Prediction: data cleaning, visualization, and insights report.

Steps (matching the project proposal):
  1. Load and audit the raw data
  2. Clean: duplicates, data types, missing values, outliers
  3. Produce 7 planned figures (saved as PNG)
  4. Write a short insights report (Markdown) built from computed numbers

Usage:
  python loan_analysis.py --data data/train.csv --out outputs
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save figures to disk; no display needed
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

SKEWED_INCOME_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
MODE_COLS = ["Gender", "Married", "Self_Employed", "Dependents"]
MEDIAN_COLS = ["LoanAmount", "Loan_Amount_Term"]


# ---------------------------------------------------------------------------
# 1. Load and audit
# ---------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].str.strip()
    return df


def audit(df, log):
    log.append(f"Raw shape: {df.shape[0]} rows x {df.shape[1]} columns")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    log.append("Missing values before cleaning:")
    log.append(missing.to_string() if len(missing) else "  none")
    log.append(f"Exact duplicate rows: {df.duplicated().sum()}")
    log.append(f"Duplicate Loan_IDs: {df['Loan_ID'].duplicated().sum()}")


# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------
def sensitivity_check(raw, log):
    """Compare approval rate under two Credit_History treatments.

    The proposal's main choice keeps missing history as 'Unknown'.
    Alternatives: (a) drop rows with missing history, (b) fill with mode (1).
    If the overall approval rate moves a lot between these, report it.
    """
    status = (raw["Loan_Status"] == "Y").astype(int)
    rate_all = status.mean()
    known = raw["Credit_History"].notna()
    rate_drop = status[known].mean()
    mode_val = raw.loc[known, "Credit_History"].mode().iloc[0]
    filled = raw["Credit_History"].fillna(mode_val)
    log.append("Sensitivity check on Credit_History missing values:")
    log.append(f"  Rows kept if dropped: {known.sum()} of {len(raw)} ({known.mean():.1%})")
    log.append(f"  Approval rate, all rows (Unknown kept as own group): {rate_all:.1%}")
    log.append(f"  Approval rate, rows with known history only:          {rate_drop:.1%}")
    log.append(f"  Approval rate, missing filled with mode ({mode_val:g}):       "
               f"{(raw.assign(Credit_History=filled)['Loan_Status'] == 'Y').mean():.1%}")
    return {
        "rows_kept_if_dropped": int(known.sum()),
        "rows_raw": len(raw),
        "rate_all": rate_all,
        "rate_known_only": rate_drop,
    }


def clean(raw, log):
    df = raw.copy()
    n_start = len(df)

    # --- Duplicates -------------------------------------------------------
    df = df.drop_duplicates()
    n_exact = n_start - len(df)
    n_id = df["Loan_ID"].duplicated().sum()
    df = df.drop_duplicates(subset="Loan_ID", keep="first")
    log.append(f"Removed {n_exact} exact duplicate rows and {n_id} duplicate Loan_IDs.")

    # --- Data types -------------------------------------------------------
    df["Dependents"] = df["Dependents"].replace("3+", "3")
    df["Loan_Status"] = df["Loan_Status"].astype("category")
    df["Loan_Approved"] = (df["Loan_Status"] == "Y").astype(int)

    # Credit_History is 1.0/0.0 in the file; make it a labeled category
    # so 'Unknown' can sit beside it without being read as a number.
    ch_map = {1.0: "Good (1)", 0.0: "Bad (0)"}
    df["Credit_History"] = df["Credit_History"].map(ch_map)

    # --- Missing values ---------------------------------------------------
    miss_before = df.isna().sum()
    for col in MODE_COLS:
        if miss_before[col]:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    for col in MEDIAN_COLS:
        if miss_before[col]:
            df[col] = df[col].fillna(df[col].median())
    # Missing credit history is kept as its own group, per the proposal
    df["Credit_History"] = df["Credit_History"].fillna("Unknown")

    # Dependents: keep as integer 0-3 after filling
    df["Dependents"] = df["Dependents"].astype(int)

    log.append("Imputation applied:")
    for col in MODE_COLS + MEDIAN_COLS + ["Credit_History"]:
        if miss_before.get(col, 0) or col == "Credit_History":
            how = "mode" if col in MODE_COLS else "median" if col in MEDIAN_COLS else "'Unknown' category"
            log.append(f"  {col}: {miss_before.get(col, 0)} missing -> {how}")

    # --- Outliers: flag, keep, and log-transform -------------------------
    log.append("Outliers (1.5 x IQR rule), flagged and kept:")
    for col in SKEWED_INCOME_COLS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        flag = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        df[f"{col}_outlier"] = flag
        df[f"log_{col}"] = np.log1p(df[col])
        log.append(f"  {col}: {flag.sum()} flagged")

    log.append(f"Rows after cleaning: {len(df)} (started with {n_start})")
    return df


# ---------------------------------------------------------------------------
# 3. Figures (Figure numbers match the proposal)
# ---------------------------------------------------------------------------
def approval_rate_table(df, col):
    return df.groupby(col, observed=True)["Loan_Approved"].agg(["mean", "size"])


def bar_approval(df, col, title, path):
    table = approval_rate_table(df, col)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(table.index.astype(str), table["mean"], color="steelblue")
    for bar, (rate, n) in zip(bars, table.itertuples(index=False)):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                f"{rate:.0%}\n(n={n})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Approval rate")
    ax.set_xlabel(col)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(df, fig_dir):
    paths = {}

    # Figure 1: target distribution
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.countplot(x="Loan_Status", data=df, order=["Y", "N"], hue="Loan_Status",
                  palette=["seagreen", "indianred"], legend=False, ax=ax)
    ax.set_title("Figure 1: Loan status distribution")
    ax.set_xlabel("Loan_Status (Y = approved, N = rejected)")
    fig.tight_layout()
    paths[1] = os.path.join(fig_dir, "fig01_loan_status.png")
    fig.savefig(paths[1], dpi=150)
    plt.close(fig)

    # Figure 2: income raw vs log
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.histplot(df["ApplicantIncome"], kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title("ApplicantIncome (raw)")
    sns.histplot(df["log_ApplicantIncome"], kde=True, ax=axes[1], color="darkorange")
    axes[1].set_title("ApplicantIncome (log1p)")
    axes[1].set_xlabel("log(1 + ApplicantIncome)")
    fig.suptitle("Figure 2: Income distribution before and after log transform")
    fig.tight_layout()
    paths[2] = os.path.join(fig_dir, "fig02_income_raw_vs_log.png")
    fig.savefig(paths[2], dpi=150)
    plt.close(fig)

    # Figure 3: loan amount by property area
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="Property_Area", y="LoanAmount", data=df,
                order=["Urban", "Semiurban", "Rural"], hue="Property_Area",
                palette="Set2", legend=False, ax=ax)
    ax.set_title("Figure 3: Loan amount by property area")
    ax.set_ylabel("LoanAmount (thousands)")
    fig.tight_layout()
    paths[3] = os.path.join(fig_dir, "fig03_loanamount_by_area.png")
    fig.savefig(paths[3], dpi=150)
    plt.close(fig)

    # Figure 4: approval rate by credit history
    paths[4] = os.path.join(fig_dir, "fig04_approval_by_credit_history.png")
    bar_approval(df, "Credit_History", "Figure 4: Approval rate by credit history", paths[4])

    # Figure 5: approval rate by property area
    paths[5] = os.path.join(fig_dir, "fig05_approval_by_property_area.png")
    bar_approval(df, "Property_Area", "Figure 5: Approval rate by property area", paths[5])

    # Figure 6: correlation heatmap of numeric features
    num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                "Loan_Amount_Term", "Dependents", "Loan_Approved"]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, ax=ax)
    ax.set_title("Figure 6: Correlation among numeric features")
    fig.tight_layout()
    paths[6] = os.path.join(fig_dir, "fig06_correlation_heatmap.png")
    fig.savefig(paths[6], dpi=150)
    plt.close(fig)

    # Figure 7: income vs loan amount, colored by status
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x="ApplicantIncome", y="LoanAmount", hue="Loan_Status",
                    data=df, alpha=0.6, palette={"Y": "seagreen", "N": "indianred"}, ax=ax)
    ax.set_title("Figure 7: Applicant income vs loan amount by status")
    ax.set_xlabel("ApplicantIncome")
    ax.set_ylabel("LoanAmount (thousands)")
    fig.tight_layout()
    paths[7] = os.path.join(fig_dir, "fig07_income_vs_loan_by_status.png")
    fig.savefig(paths[7], dpi=150)
    plt.close(fig)

    return paths


# ---------------------------------------------------------------------------
# 4. Insights report (every number is computed, not typed in)
# ---------------------------------------------------------------------------
def write_report(df, sens, log, fig_paths, out_path):
    overall = df["Loan_Approved"].mean()
    ch = approval_rate_table(df, "Credit_History")
    area = approval_rate_table(df, "Property_Area")
    ch_gap = ch["mean"].max() - ch["mean"].min()
    area_gap = area["mean"].max() - area["mean"].min()

    num = df[["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Dependents"]]
    raw_corr = num.corr()
    abs_corr = raw_corr.abs().to_numpy(copy=True)
    np.fill_diagonal(abs_corr, 0)
    i, j = np.unravel_index(np.argmax(abs_corr), abs_corr.shape)
    top_pair = (raw_corr.index[i], raw_corr.columns[j], raw_corr.iloc[i, j])

    med_income = df.groupby("Loan_Status", observed=True)["ApplicantIncome"].median()
    med_loan_area = df.groupby("Property_Area")["LoanAmount"].median()

    lines = []
    lines.append("# Insights Report: Loan Approval Dataset\n")
    lines.append("## Scope and caveats")
    lines.append(f"- Analysis covers {len(df)} applications after cleaning.")
    lines.append("- Descriptive analysis only. Associations are not causal, and the data's origin is undocumented.")
    lines.append("- Gender, marital status, and property area are sensitive attributes. Differences across these "
                 "groups are not evidence about how applicants should be treated.\n")

    lines.append("## Key findings")
    lines.append(f"1. **Approval rate is {overall:.1%}** (Figure 1). "
                 f"The target is {'imbalanced' if abs(overall - 0.5) > 0.2 else 'reasonably balanced'}.")
    lines.append("2. **Credit history is associated with approval** (Figure 4):")
    for idx, row in ch.iterrows():
        lines.append(f"   - {idx}: {row['mean']:.1%} approved (n={int(row['size'])})")
    lines.append(f"   - Gap between highest and lowest group: {ch_gap:.1%}.")
    lines.append(f"3. **Property area differences** (Figure 5), gap between highest and lowest: {area_gap:.1%}:")
    for idx, row in area.iterrows():
        lines.append(f"   - {idx}: {row['mean']:.1%} approved (n={int(row['size'])})")
    verdict = ("Credit history shows the larger gap, which is consistent with it being the stronger signal here."
               if ch_gap > area_gap else
               "Property area shows the larger gap, which contradicts the expectation that credit history dominates. "
               "Check the data before trusting this.")
    lines.append(f"   - Comparison: {verdict}")
    lines.append(f"4. **Income separates approvals only partly** (Figure 7). Median ApplicantIncome: "
                 f"approved {med_income.get('Y', float('nan')):,.0f}, rejected {med_income.get('N', float('nan')):,.0f}. "
                 "Check the scatter plot for overlap before claiming separation.")
    lines.append(f"5. **Strongest numeric correlation** (Figure 6): {top_pair[0]} and {top_pair[1]} "
                 f"at r = {top_pair[2]:.2f}.")
    lines.append(f"6. **Median LoanAmount by area** (Figure 3): " +
                 ", ".join(f"{k} {v:,.0f}" for k, v in med_loan_area.items()) + " (thousands).\n")

    lines.append("## Sensitivity to missing Credit_History")
    lines.append(f"- Rows kept if dropped: {sens['rows_kept_if_dropped']} of {sens['rows_raw']} raw rows. "
                 f"Known-history approval rate: {sens['rate_known_only']:.1%}, versus "
                 f"{sens['rate_all']:.1%} with 'Unknown' kept.")
    lines.append("- If these two numbers differ meaningfully, the Credit_History conclusion depends on how "
                 "missing values were handled. Say so in the final report.\n")

    lines.append("## Cleaning log")
    lines.append("```")
    lines.extend(log)
    lines.append("```\n")

    lines.append("## Figures")
    for n, p in fig_paths.items():
        lines.append(f"- Figure {n}: `{os.path.relpath(p, os.path.dirname(out_path))}`")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    log = []
    raw = load_data(args.data)
    audit(raw, log)
    sens = sensitivity_check(raw, log)
    df = clean(raw, log)

    df.to_csv(os.path.join(args.out, "clean_train.csv"), index=False)
    fig_paths = make_figures(df, fig_dir)
    write_report(df, sens, log, fig_paths, os.path.join(args.out, "insights_report.md"))

    print("\n".join(log))
    print(f"\nDone. Figures in {fig_dir}, report at {os.path.join(args.out, 'insights_report.md')}")


if __name__ == "__main__":
    main()