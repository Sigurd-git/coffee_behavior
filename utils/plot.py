import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd


def load_data(experiment_type, data_path="data", patterns=None):
    """
    加载实验数据

    Parameters:
    -----------
    experiment_type : str
        实验类型，例如"nback"、"stroop"等
    data_path : str
        数据文件路径
    patterns : dict
        会话文件的匹配模式，格式为{1: pattern1, 2: pattern2}

    Returns:
    --------
    dict : 包含两个会话数据的字典，格式为{1: [files1], 2: [files2]}
    """
    if patterns is None:
        # 默认的文件匹配模式
        patterns = {
            1: f"{experiment_type}_*_session1.csv",
            2: f"{experiment_type}_*_session2.csv",
        }

    session_files = {}
    for session, pattern in patterns.items():
        full_pattern = os.path.join(data_path, pattern)
        session_files[session] = glob.glob(full_pattern)

    return session_files


def calculate_descriptive_stats(df, group_vars=None):
    """
    计算描述性统计量

    Parameters:
    -----------
    df : DataFrame
        数据框
    group_vars : list
        分组变量，默认为["session", "condition"]

    Returns:
    --------
    DataFrame : 描述性统计量
    """
    if group_vars is None:
        group_vars = ["session", "condition"]

    mean_stats = df.groupby(group_vars)["mean_rt"].agg(["mean"]).round(2).reset_index()
    std_stats = df.groupby(group_vars)["mean_rt"].agg(["std"]).round(2).reset_index()
    count_stats = (
        df.groupby(group_vars)["mean_rt"].agg(["count"]).round(2).reset_index()
    )

    desc_stats = pd.merge(mean_stats, std_stats, on=group_vars)
    desc_stats = pd.merge(desc_stats, count_stats, on=group_vars)

    return desc_stats


def paired_t_test(df, condition_var="condition", condition_value=None):
    """
    进行配对t检验

    Parameters:
    -----------
    df : DataFrame
        数据框
    condition_var : str
        条件变量名
    condition_value : any
        条件变量值，如果为None则不筛选

    Returns:
    --------
    dict : 包含统计结果的字典
    """
    # 筛选数据
    if condition_value is not None:
        test_df = df[df[condition_var] == condition_value]
    else:
        test_df = df

    # 找到两个session都有数据的被试
    participants = set(test_df[test_df["session"] == 1]["participant_id"]) & set(
        test_df[test_df["session"] == 2]["participant_id"]
    )

    if not participants:
        return {
            "n_subjects": 0,
            "t_stat": None,
            "p_val": None,
            "session1_mean": None,
            "session2_mean": None,
        }

    # 准备配对数据
    session1_rt = []
    session2_rt = []

    for pid in participants:
        s1_data = test_df[
            (test_df["session"] == 1) & (test_df["participant_id"] == pid)
        ]
        s2_data = test_df[
            (test_df["session"] == 2) & (test_df["participant_id"] == pid)
        ]

        if not s1_data.empty and not s2_data.empty:
            session1_rt.append(s1_data["mean_rt"].iloc[0])
            session2_rt.append(s2_data["mean_rt"].iloc[0])

    # 进行统计检验
    if len(session1_rt) > 1:
        t_stat, p_val = stats.ttest_rel(session1_rt, session2_rt)
        return {
            "n_subjects": len(session1_rt),
            "t_stat": t_stat,
            "p_val": p_val,
            "session1_mean": np.mean(session1_rt),
            "session2_mean": np.mean(session2_rt),
        }
    else:
        return {
            "n_subjects": len(session1_rt),
            "t_stat": None,
            "p_val": None,
            "session1_mean": np.mean(session1_rt) if session1_rt else None,
            "session2_mean": np.mean(session2_rt) if session2_rt else None,
        }


def plot_bar_comparison(
    df,
    x_var,
    y_var="mean_rt",
    hue_var="session",
    title=None,
    xlabel=None,
    ylabel=None,
    palette=None,
    ax=None,
    show_values=True,
):
    """
    绘制对比条形图

    Parameters:
    -----------
    df : DataFrame
        数据框
    x_var : str
        x轴变量
    y_var : str
        y轴变量
    hue_var : str
        分组变量
    title : str
        图表标题
    xlabel : str
        x轴标签
    ylabel : str
        y轴标签
    palette : list
        配色方案
    ax : Axes
        matplotlib坐标轴对象
    show_values : bool
        是否显示数值标签

    Returns:
    --------
    Axes : matplotlib坐标轴对象
    """
    if palette is None:
        palette = ["lightblue", "lightgreen"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # 绘制条形图
    sns.barplot(
        x=x_var,
        y=y_var,
        hue=hue_var,
        data=df,
        palette=palette,
        ax=ax,
        ci=68,
    )

    # 设置标题和标签
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # 添加数值标签
    if show_values:
        for i, session in enumerate([1, 2]):
            session_df = df[df[hue_var] == session]
            for j, val in enumerate(sorted(df[x_var].unique())):
                sub_df = session_df[session_df[x_var] == val]
                if not sub_df.empty:
                    mean_val = sub_df[y_var].mean()
                    x = j + (i - 0.5) * 0.35
                    ax.text(x, mean_val, f"{mean_val:.1f}", ha="center", va="bottom")

    # Set legend
    # ax.legend(title="Experiment Phase", labels=["Pre-coffee", "Post-coffee"])

    return ax


def add_significance_markers(
    ax, x_positions, p_value, height, dy=0.02, text_offset=0.01
):
    """
    Add significance markers (*, **, ***) to bar plots based on p-values

    Parameters:
    -----------
    ax : matplotlib axis
        The axis to add markers to
    x_positions : list or tuple
        The x positions of the bars to connect with a line (e.g., [0, 1])
    p_value : float
        p-value from statistical test
    height : float
        Height at which to place the significance line
    dy : float
        Vertical offset for the line
    text_offset : float
        Vertical offset for the text
    """
    if p_value <= 0.001:
        marker = "***"
    elif p_value <= 0.01:
        marker = "**"
    elif p_value <= 0.05:
        marker = "*"
    else:
        return

    # Draw a line between the bars
    x1, x2 = x_positions
    y = height + dy
    ax.plot([x1, x2], [y, y], "k-", linewidth=1)

    # Add the significance marker
    ax.text(
        (x1 + x2) / 2, y + text_offset, marker, ha="center", va="bottom", color="black"
    )
