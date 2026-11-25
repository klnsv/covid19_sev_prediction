"""Step 04: does cell-type composition differ between control, mild and severe?

Reads:  data/processed/patient_meta.tsv, data/processed/features_composition.tsv
Writes: results/composition/composition_tests.tsv
        results/composition/figures/04_composition_box.png

Tests run on CLR values; medians and the figure use proportions because they read
more easily. BH correction covers all 17 cell types within each comparison.

Run from the repo root: python scripts/04_composition_tests.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

import common

GROUPS = ["control", "mild", "severe"]
GROUP_COLORS = {"control": "#2a78d6", "mild": "#1baf7a", "severe": "#eda100"}
N_TYPES_IN_FIGURE = 6

OUT_DIR = common.RESULTS / "composition"
FIG_DIR = OUT_DIR / "figures"


def higher_side(values_a, values_b, name_a, name_b):
    median_a, median_b = np.median(values_a), np.median(values_b)
    if median_a == median_b:
        return "no_difference"
    return f"higher_in_{name_a if median_a > median_b else name_b}"


def run_tests(clr):
    control, mild, severe = (clr[clr["group"] == group]["value"].to_numpy() for group in GROUPS)
    covid = np.concatenate([mild, severe])

    medians = {"control": np.median(control), "mild": np.median(mild), "severe": np.median(severe)}
    highest_group = max(medians, key=medians.get)

    kw_stat, kw_p = kruskal(control, mild, severe)
    mild_stat, mild_p = mannwhitneyu(severe, mild, alternative="two-sided")
    covid_stat, covid_p = mannwhitneyu(covid, control, alternative="two-sided")

    return [
        ("kw_control_mild_severe", "kruskal_wallis", kw_stat, kw_p, f"highest_in_{highest_group}"),
        ("mild_vs_severe", "mann_whitney_u", mild_stat, mild_p, higher_side(severe, mild, "severe", "mild")),
        ("control_vs_covid", "mann_whitney_u", covid_stat, covid_p, higher_side(covid, control, "covid", "control")),
    ]


def build_test_table(data, cell_types):
    n_per_group = {group: int((data["group"] == group).sum()) for group in GROUPS}

    rows = []
    for cell_type in cell_types:
        clr = data[["group"]].assign(value=data[f"clr__{cell_type}"])
        median_prop = data.groupby("group")[f"prop__{cell_type}"].median()

        for comparison, test, statistic, pvalue, direction in run_tests(clr):
            rows.append({
                "ct": cell_type,
                "comparison": comparison,
                "test": test,
                "statistic": statistic,
                "pvalue": pvalue,
                "median_control": median_prop["control"],
                "median_mild": median_prop["mild"],
                "median_severe": median_prop["severe"],
                "n_control": n_per_group["control"],
                "n_mild": n_per_group["mild"],
                "n_severe": n_per_group["severe"],
                "direction": direction,
            })

    table = pd.DataFrame(rows)
    table["qvalue_bh"] = np.nan
    for comparison, tests in table.groupby("comparison"):
        table.loc[tests.index, "qvalue_bh"] = multipletests(tests["pvalue"], method="fdr_bh")[1]

    columns = ["ct", "comparison", "test", "statistic", "pvalue", "qvalue_bh",
               "median_control", "median_mild", "median_severe",
               "n_control", "n_mild", "n_severe", "direction"]
    return table[columns]


def plot_top_cell_types(data, table, n_tested, out_path):
    top = (table[table["comparison"] == "mild_vs_severe"]
           .sort_values(["qvalue_bh", "pvalue"])
           .head(N_TYPES_IN_FIGURE))

    rng = np.random.default_rng(common.SEED)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for ax, (_, row) in zip(axes.ravel(), top.iterrows()):
        values = [100 * data.loc[data["group"] == group, f"prop__{row['ct']}"].to_numpy() for group in GROUPS]
        boxes = ax.boxplot(values, widths=0.55, showfliers=False, patch_artist=True,
                           medianprops={"color": "#333333"})
        for box, group in zip(boxes["boxes"], GROUPS):
            box.set_facecolor(GROUP_COLORS[group])
            box.set_alpha(0.35)
        for position, (group, points) in enumerate(zip(GROUPS, values), start=1):
            jitter = rng.uniform(-0.15, 0.15, len(points))
            ax.scatter(position + jitter, points, s=14, color=GROUP_COLORS[group], linewidths=0, zorder=3)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([f"{group}\nn = {len(points)}" for group, points in zip(GROUPS, values)], fontsize=9)
        ax.set_title(f"{row['ct'].replace('_', ' ')}\nmild vs severe q = {row['qvalue_bh']:.3f}", fontsize=10)
        ax.set_ylabel("% of donor's cells")
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)

    fig.suptitle(f"Top {N_TYPES_IN_FIGURE} of {n_tested} cell types by mild vs severe q-value "
                 "(Mann-Whitney U on CLR, BH across all types; each point = one patient)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(common.PATIENT_META, sep="\t")
    composition = pd.read_csv(common.FEATURES_COMPOSITION, sep="\t")
    data = meta.merge(composition, on="donor_id", validate="one_to_one")
    data = data[data["group"].isin(GROUPS)]

    cell_types = [col.removeprefix("clr__") for col in composition.columns if col.startswith("clr__")]
    print(f"{len(data)} donors, {len(cell_types)} cell types")

    table = build_test_table(data, cell_types)
    table_path = OUT_DIR / "composition_tests.tsv"
    table.to_csv(table_path, sep="\t", index=False)
    print(f"Wrote {table_path}")

    figure_path = FIG_DIR / "04_composition_box.png"
    plot_top_cell_types(data, table, len(cell_types), figure_path)
    print(f"Wrote {figure_path}")

    print("\nTests with q < 0.05")
    significant = table[table["qvalue_bh"] < 0.05].sort_values(["comparison", "qvalue_bh"])
    print(significant[["comparison", "ct", "pvalue", "qvalue_bh", "direction"]].to_string(index=False))

    print("\nMild vs severe, smallest q-values")
    mild_severe = table[table["comparison"] == "mild_vs_severe"].sort_values("qvalue_bh")
    print(mild_severe[["ct", "pvalue", "qvalue_bh", "median_mild", "median_severe", "direction"]]
          .head(N_TYPES_IN_FIGURE).to_string(index=False))


if __name__ == "__main__":
    main()
