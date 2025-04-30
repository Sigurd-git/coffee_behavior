import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
from scipy import stats
from utils.plot import paired_t_test, plot_bar_comparison
from utils.tasks import extract_bart
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statannotations.Annotator import Annotator


def analyze_bart():
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    bart_acc_config = {
        "experiment_type": "bart",
    }

    # Extract earnings data
    exp_df = extract_bart(bart_acc_config)
    exp_df = exp_df[
        ~exp_df["participant_id"].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 16, 19, 25])
    ]
    if exp_df.empty:
        print("No valid BART data found")
        return

    # control_df = earnings_df[earnings_df["participant_id"] == 8]

    # 2 x 2 anova 检验condition和session对total_earned的影响
    # 使用 statsmodels 进行anova
    print("\n==== ANOVA earnings ====")
    model = ols("total_earned ~ group * session", data=exp_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    print("\n==== ANOVA pumps ====")
    model = ols("pumps ~ group * session", data=exp_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    print("\n==== ANOVA explosion_rate ====")
    model = ols("explosion_rate ~ group * session", data=exp_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # Create visualizations
    earnings_config = {
        "title": "BART Task Total Earnings",
        "ylabel": "Total Earnings",
        "measure": "total_earned",
        "output_file": "bart_total_earnings.png",
    }

    pumps_config = {
        "title": "BART Task Pumps",
        "ylabel": "Pumps",
        "measure": "pumps",
        "output_file": "bart_pumps.png",
    }

    explosion_rate_config = {
        "title": "BART Task Explosion Rate",
        "ylabel": "Explosion Rate",
        "measure": "explosion_rate",
        "output_file": "bart_explosion_rate.png",
    }
    plot_configs = [earnings_config, pumps_config, explosion_rate_config]
    print("\n==== Analyzing experimental group data (excluding subject 8) ====")
    analyze_group_earnings(exp_df, output_dir, plot_configs)


def analyze_group_earnings(df, output_dir, plot_configs):
    """
    Analyze BART earnings for a specific group

    Parameters:
    -----------
    df : DataFrame
        DataFrame containing earnings data for the group
    output_dir : str
        Directory to save output files
    """
    if df.empty:
        print("No valid data for this group")
        return

    # 计算参与者人数
    participant_count = df["participant_id"].nunique()
    print(f"This group contains {participant_count} participants")
    participant_list = df["participant_id"].astype(str).unique().tolist()
    for config in plot_configs:
        ax = plot_bar_comparison(
            df,
            x_var="group",
            y_var=config["measure"],
            title=config["title"],
            ylabel=config["ylabel"],
        )
        # 添加显著性标记
        x_var = "group"
        y_var = config.get("y_var", config["measure"])  # 使用正确的y变量名
        hue_var = config.get("hue_var", "session")
        x_values = df[x_var].unique()
        hue_values = df[hue_var].unique()
        box_pairs = []
        combinations = []
        for x in x_values:
            for h in hue_values:
                combinations.append((x, h))
        for i in range(len(combinations)):
            for j in range(i + 1, len(combinations)):
                box_pairs.append((combinations[i], combinations[j]))
        annotator = Annotator(
            ax,
            box_pairs,
            data=df,
            x=x_var,
            y=y_var,
            order=x_values,
            hue=hue_var,
        )
        annotator.configure(
            test="t-test_paired",
            text_format="star",
            loc="inside",
            hide_non_significant=True,
        )
        annotator.apply_and_annotate()
        ax.figure.savefig(
            os.path.join(
                output_dir,
                "-".join(participant_list) + "_" + config["output_file"],
            )
        )


def analyze_bart_high_mid_low():
    """
    Analyze BART earnings data and compare between sessions
    """
    print("Starting analysis of BART task earnings data...")

    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    bart_acc_config = {
        "experiment_type": "bart",
        "extract_function": extract_bart,
        "conditions": ["overall"],  # BART只有一个总体条件
    }

    # Extract earnings data
    earnings_df = extract_bart(bart_acc_config)

    if earnings_df.empty:
        print("No valid BART data found")
        return

    # 将数据分为 低，中，高
    low_df = earnings_df[
        earnings_df["participant_id"].isin(
            [20, 11, 14, 12, 16, 22, 26, 24, 18, 21, 25, 30, 31]
        )
    ]
    low_df["condition"] = "low"
    mid_df = earnings_df[earnings_df["participant_id"].isin([13, 28, 29, 33])]
    mid_df["condition"] = "mid"
    high_df = earnings_df[earnings_df["participant_id"].isin([23, 17, 15, 19, 32, 27])]
    high_df["condition"] = "high"
    exp_df = pd.concat([low_df, mid_df, high_df])
    print("\n==== Analyzing experimental group data ====")

    # Create visualizations
    earnings_config = {
        "title": "BART Task Total Earnings",
        "ylabel": "Total Earnings",
        "measure": "total_earned",
        "output_file": "bart_total_earnings.png",
    }

    pumps_config = {
        "title": "BART Task Pumps",
        "ylabel": "Pumps",
        "measure": "pumps",
        "output_file": "bart_pumps.png",
    }

    explosion_rate_config = {
        "title": "BART Task Explosion Rate",
        "ylabel": "Explosion Rate",
        "measure": "explosion_rate",
        "output_file": "bart_explosion_rate.png",
    }
    plot_configs = [earnings_config, pumps_config, explosion_rate_config]

    analyze_group_earnings(exp_df, output_dir, plot_configs)


def analyze_bart_learning():
    print("Starting analysis of BART task earnings data...")

    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    bart_acc_config = {
        "experiment_type": "bart",
        "extract_function": extract_bart,
        "conditions": ["overall"],  # BART只有一个总体条件
    }

    # Extract earnings data
    data_df = extract_bart(bart_acc_config)

    if data_df.empty:
        print("No valid BART data found")
        return

    exp_df = data_df[~data_df["participant_id"].isin([0, 1, 8])]
    exp_df = pd.melt(
        exp_df,
        id_vars=[
            "participant_id",
            "session",
        ],
        value_vars=["variance_first_half", "variance_second_half"],
        var_name="condition",
        value_name="variance",
    )
    # 2x 2 anova 检验session和condition对variance的影响
    # 使用 statsmodels 进行ancova ,temporal作为连续变量
    exp_df["temporal"] = exp_df.apply(
        lambda row: 1
        if row["session"] == 1 and row["condition"] == "variance_first_half"
        else (
            2
            if row["session"] == 2 and row["condition"] == "variance_first_half"
            else (
                3
                if row["session"] == 1 and row["condition"] == "variance_second_half"
                else 4
            )
        ),
        axis=1,
    )
    # 使用 statsmodels 进行方差分析
    model = ols("variance ~ session * condition", data=exp_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # condition为variance_first_half,session为1时，设置temporal=1, session为2时，设置temporal=2
    # condition为variance_second_half,session为1时，设置temporal=3, session为2时，设置temporal=4
    # 使用 statsmodels 进行anova ,temporal作为分类变量
    exp_df["temporal"] = exp_df.apply(
        lambda row: str(1)
        if row["session"] == 1 and row["condition"] == "variance_first_half"
        else (
            str(2)
            if row["session"] == 2 and row["condition"] == "variance_first_half"
            else (
                str(3)
                if row["session"] == 1 and row["condition"] == "variance_second_half"
                else str(4)
            )
        ),
        axis=1,
    )

    model = ols("variance ~ session * temporal", data=exp_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    print("\n==== Analyzing experimental group data ====")
    plot_configs = [
        {
            "title": "BART Task Variance",
            "ylabel": "Variance",
            "measure": "variance",
            "output_file": "bart_variance.png",
        }
    ]
    analyze_group_earnings(exp_df, output_dir, plot_configs)


if __name__ == "__main__":
    analyze_bart()
    # analyze_bart_learning()
