import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.plot import (
    plot_bar_comparison,
    paired_t_test,
    add_significance_markers,
)
from utils.tasks import (
    extract_nback,
)

def analyze_experiment_accuracy(
    experiment_config, exclude_participant=None, only_participant=None
):
    """
    通用实验正确率分析主函数

    Parameters:
    -----------
    experiment_config : dict
        实验配置字典，包含实验类型、数据提取函数、分析参数等

    Returns:
    --------
    dict : 分析结果字典
    """
    results = {}

    # 2. 提取数据
    extract_func = experiment_config["extract_function"]
    df = extract_func(experiment_config)
    if isinstance(df, tuple):
        df, original_df = df
    # 3. 如有需要，根据被试ID过滤数据
    if exclude_participant is not None:
        if isinstance(exclude_participant, list):
            df = df[~df["participant_id"].isin(exclude_participant)]
        else:
            df = df[df["participant_id"] != exclude_participant]

    if only_participant is not None:
        if isinstance(only_participant, list):
            df = df[df["participant_id"].isin(only_participant)]
        else:
            df = df[df["participant_id"] == only_participant]

    # 如果过滤后没有数据，则跳过当前实验
    if df.empty:
        print(
            f"Experiment {experiment_config['experiment_type']} has no valid data after filtering"
        )
        return results

    # 6. 可视化
    if experiment_config.get("plot", True):
        plot_config = experiment_config.get("plot_config", {})
        row_var = plot_config.get("row_var", None)
        if row_var is not None:
            row_values = df[row_var].unique()
            fig, axes = plt.subplots(
                figsize=plot_config.get("figsize", (8, 5)),
                nrows=len(row_values),
                ncols=1,
            )
        else:
            row_values = [None]
            fig, ax = plt.subplots(figsize=plot_config.get("figsize", (8, 5)))
            axes = [ax]
        title = plot_config.get(
            "title",
            f"{experiment_config['experiment_type']} Task Accuracy Comparison",
        )
        fig.suptitle(title)
        for row_value, ax in zip(row_values, axes):
            if row_value is not None:
                df_row = df[df[row_var] == row_value]
            else:
                df_row = df
            plot_bar_comparison(
                df_row,
                x_var=plot_config.get("x_var", "condition"),
                y_var=plot_config.get("y_var", "accuracy"),  # 使用accuracy作为y轴变量
                hue_var=plot_config.get("hue_var", "session"),
                xlabel=plot_config.get("xlabel", "Condition"),
                ylabel=plot_config.get("ylabel", "Accuracy (%)"),
                palette=plot_config.get("palette", ["lightblue", "lightgreen"]),
                ax=ax,
                show_values=plot_config.get("show_values", True),
            )
            ax.set_title(row_value)
        plt.tight_layout()

        # 保存图表
        output_dir = experiment_config.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            plot_config.get(
                "output_file",
                f"{experiment_config['experiment_type']}_accuracy_comparison.png",
            ),
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    return results


def analyze_combined_accuracy(
    experiments_config, exclude_participant=None, only_participant=None
):
    """
    分析多个实验的正确率并进行对比

    Parameters:
    -----------
    experiments_config : list
        实验配置列表，每个元素为一个实验配置字典
    exclude_participant : int or list
        要排除的被试ID，如果为None则不排除任何被试
    only_participant : int or list
        只分析特定被试ID，如果为None则分析所有被试
    """

    # 处理所有实验
    all_results = {}

    for config in experiments_config:
        # 4. 分析实验
        results = analyze_experiment_accuracy(
            config, exclude_participant, only_participant
        )
        all_results[config["experiment_type"]] = results

    return all_results


# 示例用法
if __name__ == "__main__":
    # N-back实验配置
    nback_acc_config = {
        "experiment_type": "nback",
        "extract_function": extract_nback,
        "plot_config": {
            "x_var": "condition",
            "row_var": "group",
            "title": "N-back Task Accuracy Comparison",
            "xlabel": "Condition",
            "ylabel": "Accuracy (%)",
            "output_file": "nback_accuracy_comparison_standalone.png",
        },
        "output_dir": "output",
    }

    # Stroop实验配置
    # stroop_acc_config = {
    #     "experiment_type": "stroop",
    #     "extract_function": extract_stroop,
    #     "plot_config": {
    #         "x_var": "condition",
    #         "row_var": "group",
    #         "title": "Stroop Task Accuracy Comparison",
    #         "xlabel": "Session",
    #         "ylabel": "Accuracy (%)",
    #         "output_file": "stroop_accuracy_comparison.png",
    #     },
    #     "output_dir": "output",
    # }

    # 创建输出目录
    os.makedirs("output", exist_ok=True)

    # Define participants to exclude
    exclude_ids = [0, 1, 2, 4, 8]

    print("==== This script now only contains N-back accuracy configuration ====")
    print("==== N-back accuracy analysis logic is in analyze_nback.py ====")
    print("==== Stroop accuracy analysis logic is in analyze_stroop.py ====")

    # Removed analysis calls as the functions are removed
    # print("==== Analyzing experimental group data (excluding specified IDs) ====")
    # config_list = [nback_acc_config] # Only nback remains conceptually

    # exp_results = analyze_combined_accuracy( # Function removed
    #     config_list,
    #     exclude_participant=exclude_ids,
    # )

    # print("\nN-back Accuracy configuration remains, but analysis functions removed.")
