import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
from scipy import stats
from utils.plot import add_significance_markers
from utils.tasks import extract_bart


def paired_t_test(df, measure):
    """
    Perform paired t-test on earnings data

    Parameters:
    -----------
    df : DataFrame
        DataFrame containing earnings data
    measure : str
        Column name for the measure to test

    Returns:
    --------
    dict : Dictionary containing test results
    """
    # Find participants who completed both sessions
    participants = set(df[df["session"] == 1]["participant_id"]) & set(
        df[df["session"] == 2]["participant_id"]
    )

    if not participants:
        return {
            "n_subjects": 0,
            "t_stat": None,
            "p_val": None,
            "session1_mean": None,
            "session2_mean": None,
        }

    # Prepare paired data
    session1_data = []
    session2_data = []

    for pid in participants:
        s1_data = df[(df["session"] == 1) & (df["participant_id"] == pid)]
        s2_data = df[(df["session"] == 2) & (df["participant_id"] == pid)]

        if not s1_data.empty and not s2_data.empty:
            session1_data.append(s1_data[measure].iloc[0])
            session2_data.append(s2_data[measure].iloc[0])

    # Perform statistical test
    if len(session1_data) > 1:
        t_stat, p_val = stats.ttest_rel(session1_data, session2_data)
        return {
            "n_subjects": len(session1_data),
            "t_stat": t_stat,
            "p_val": p_val,
            "session1_mean": np.mean(session1_data),
            "session2_mean": np.mean(session2_data),
        }
    else:
        return {
            "n_subjects": len(session1_data),
            "t_stat": None,
            "p_val": None,
            "session1_mean": np.mean(session1_data) if session1_data else None,
            "session2_mean": np.mean(session2_data) if session2_data else None,
        }


def plot_bart_earnings(df, measure, title, ylabel, output_file=None):
    """
    Plot BART earnings comparison between sessions

    Parameters:
    -----------
    df : DataFrame
        DataFrame containing earnings data
    measure : str
        Column name for the measure to plot
    title : str
        Plot title
    ylabel : str
        Y-axis label
    output_file : str
        Path to save the output figure
    """
    plt.figure(figsize=(8, 5))

    # Calculate descriptive statistics for plotting
    stats_df = df.groupby("session")[measure].agg(["mean", "std"]).reset_index()

    # Create bar plot
    ax = sns.barplot(
        x="condition",
        hue="session",
        y=measure,
        data=df,
        estimator=np.mean,
        ci=68,
        palette=["lightblue", "lightgreen"],
    )

    # Set labels and title
    plt.title(title)
    plt.xlabel("Session")
    plt.ylabel(ylabel)

    # Perform t-test
    results = paired_t_test(df, measure)

    # Add significance markers if applicable
    if results["p_val"] is not None and results["n_subjects"] > 1:
        max_height = df.groupby("session")[measure].mean().max() * 1.1
        x_positions = [0, 1]  # For single condition bar plot
        add_significance_markers(ax, x_positions, results["p_val"], max_height)

    plt.tight_layout()

    # Save figure if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()


def analyze_bart_earnings():
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

    # 将数据分为实验组（除8号外）和对照组（8号）
    exp_df = earnings_df[~earnings_df["participant_id"].isin([0, 1, 8])]
    # control_df = earnings_df[earnings_df["participant_id"] == 8]

    print("\n==== Analyzing experimental group data (excluding subject 8) ====")
    analyze_group_earnings(exp_df, output_dir, "exp")

    # print("\n==== Analyzing control group data (only subject 8) ====")
    # analyze_group_earnings(control_df, output_dir, "control")


def analyze_group_earnings(df, output_dir, group_name):
    """
    Analyze BART earnings for a specific group

    Parameters:
    -----------
    df : DataFrame
        DataFrame containing earnings data for the group
    output_dir : str
        Directory to save output files
    group_name : str
        Name of the group for file naming
    """
    if df.empty:
        print("No valid data for this group")
        return

    # 计算参与者人数
    participant_count = df["participant_id"].nunique()
    print(f"This group contains {participant_count} participants")

    # Perform t-tests
    measures = ["total_earned", "mean_earned_per_unexploded"]

    print("\nStatistical Test Results:")
    for measure in measures:
        results = paired_t_test(df, measure)
        measure_name = {
            "total_earned": "Total Earnings",
            "mean_earned_per_balloon": "Mean Earnings Per Balloon",
            "mean_earned_per_unexploded": "Mean Earnings Per Unexploded Balloon",
        }[measure]

        print(f"\n{measure_name}:")
        print(f"完成两个session的被试数量: {results['n_subjects']}")
        if results["t_stat"] is not None:
            print(f"t = {results['t_stat']:.3f}, p = {results['p_val']:.3f}")
            print(f"咖啡前平均值: {results['session1_mean']:.2f}")
            print(f"咖啡后平均值: {results['session2_mean']:.2f}")
        else:
            print("样本量不足，无法进行统计检验")

    # Create visualizations
    plot_configs = [
        {
            "title": f"{group_name.upper()} Group BART Task Total Earnings",
            "ylabel": "Total Earnings",
            "measure": "total_earned",
            "output_file": f"bart_{group_name}_total_earnings.png",
        },
        # {
        #     "title": f"{group_name.upper()} Group BART Task Mean Earnings Per Balloon",
        #     "ylabel": "Mean Earnings Per Balloon",
        #     "measure": "mean_earned_per_balloon",
        #     "output_file": f"bart_{group_name}_mean_per_balloon.png",
        # },
    ]

    for config in plot_configs:
        plot_bart_earnings(
            df,
            config["measure"],
            config["title"],
            config["ylabel"],
            os.path.join(output_dir, config["output_file"]),
        )


if __name__ == "__main__":
    analyze_bart_earnings()
