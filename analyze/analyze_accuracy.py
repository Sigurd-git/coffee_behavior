import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.plot import (
    load_data,
    plot_bar_comparison,
    paired_t_test,
    add_significance_markers,
)


# 提取n-back任务的正确率
def extract_nback_accuracy(session_files):
    """
    从N-back实验数据中提取正确率指标

    Parameters:
    -----------
    session_files : dict
        包含各会话数据文件的字典

    Returns:
    --------
    DataFrame : 包含被试ID、会话、条件和正确率的数据框
    """
    all_data = {
        "task": [],
        "participant_id": [],
        "session": [],
        "condition": [],
        "accuracy": [],
    }

    for session, files in session_files.items():
        for file in files:
            df = pd.read_csv(file)
            # 只选择正确的试次和目标试次
            target_trials = df[df["target"]]
            # 计算每个n-back水平的正确率
            for n_back in df["n_back"].unique():
                n_back_trials = target_trials[target_trials["n_back"] == n_back]
                correct_trials = n_back_trials[n_back_trials["correct"]]
                accuracy = len(correct_trials) / len(n_back_trials) * 100

                all_data["task"].append("n-back")
                all_data["participant_id"].append(df["participant_id"].iloc[0])
                all_data["session"].append(session)
                all_data["condition"].append(int(n_back))
                all_data["accuracy"].append(accuracy)

    return pd.DataFrame(all_data)


# 提取Stroop任务的正确率
def extract_stroop_accuracy(session_files):
    """
    从Stroop实验数据中提取正确率指标

    Parameters:
    -----------
    session_files : dict
        包含各会话数据文件的字典

    Returns:
    --------
    DataFrame : 包含被试ID、会话、条件和正确率的数据框
    """
    all_data = {
        "task": [],
        "participant_id": [],
        "session": [],
        "condition": [],
        "accuracy": [],
    }

    for session, files in session_files.items():
        for file in files:
            df = pd.read_csv(file)

            # 对于Stroop任务，从统计文件中获取总体指标
            total_data = df[df["level"] == "total"].iloc[0]
            all_data["task"].append("stroop")
            all_data["participant_id"].append(total_data["participant_id"])
            all_data["session"].append(session)
            all_data["condition"].append("total")
            all_data["accuracy"].append(total_data["正确率"])

    return pd.DataFrame(all_data)


# 提取BART任务的准确率
def extract_bart_accuracy(session_files):
    """
    Extract accuracy metrics from BART experiment data.
    For BART, accuracy is defined as (1 - proportion of exploded balloons) * 100%

    Parameters:
    -----------
    session_files : dict
        Dictionary containing session data files

    Returns:
    --------
    DataFrame : DataFrame containing participant ID, session, condition and accuracy
    """
    all_data = {
        "task": [],
        "participant_id": [],
        "session": [],
        "condition": [],
        "explosion_rate": [],
    }

    for session, files in session_files.items():
        for file in files:
            df = pd.read_csv(file)
            # remove rows with any NA values
            df = df[df["exploded"].notna()]
            # Count total trials and exploded trials
            total_trials = len(df)
            exploded_trials = len(df[df["exploded"]])

            # Calculate accuracy as 1 - explosion rate
            if total_trials > 0:
                explosion_rate = (exploded_trials / total_trials) * 100

                # Extract participant ID from filename
                file_name = os.path.basename(file)
                participant_id = int(file_name.split("_")[1])

                all_data["task"].append("BART")
                all_data["participant_id"].append(participant_id)
                all_data["session"].append(session)
                all_data["condition"].append("overall")
                all_data["explosion_rate"].append(explosion_rate)

                print(
                    f"Processed BART explosion rate for file: {file}, explosion rate: {explosion_rate:.2f}%"
                )

    return pd.DataFrame(all_data)


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

    # 1. 加载数据
    session_files = load_data(
        experiment_config["experiment_type"],
        experiment_config.get("data_path", "data"),
        experiment_config.get("file_patterns", None),
    )

    # 2. 提取数据
    extract_func = experiment_config["extract_function"]
    df = extract_func(session_files)

    # 3. 如有需要，根据被试ID过滤数据
    if exclude_participant is not None:
        df = df[df["participant_id"] != exclude_participant]

    if only_participant is not None:
        df = df[df["participant_id"] == only_participant]

    # 如果过滤后没有数据，则跳过当前实验
    if df.empty:
        print(
            f"Experiment {experiment_config['experiment_type']} has no valid data after filtering"
        )
        return results

    # 3. 描述性统计
    plot_config = experiment_config.get("plot_config", {})
    y_var = plot_config.get("y_var", "accuracy")
    group_vars = experiment_config.get("group_vars", ["session", "condition"])
    mean_stats = df.groupby(group_vars)[y_var].agg(["mean"]).round(2).reset_index()
    std_stats = df.groupby(group_vars)[y_var].agg(["std"]).round(2).reset_index()
    count_stats = df.groupby(group_vars)[y_var].agg(["count"]).round(2).reset_index()

    desc_stats = pd.merge(mean_stats, std_stats, on=group_vars)
    desc_stats = pd.merge(desc_stats, count_stats, on=group_vars)
    results["descriptive_stats"] = desc_stats

    # 4. 统计检验
    ttest_results = {}
    conditions = experiment_config.get("conditions", [None])
    condition_var = experiment_config.get("condition_var", "condition")

    for condition in conditions:
        condition_key = str(condition) if condition is not None else "overall"
        # 重用common_analysis中的paired_t_test函数，但用于accuracy变量
        test_df = df.copy()
        test_df["mean_rt"] = test_df[y_var]  # 临时重命名以适应paired_t_test函数
        ttest_results[condition_key] = paired_t_test(test_df, condition_var, condition)

    results["ttest_results"] = ttest_results

    # 5. 打印结果
    print(
        f"\n{experiment_config['experiment_type'].upper()} Task {y_var.capitalize()} Analysis Results:"
    )
    print("\nDescriptive Statistics:")
    print(desc_stats)

    print("\nStatistical Test Results:")
    for condition, result in ttest_results.items():
        if result["n_subjects"] > 0:
            condition_name = f"{condition}" if condition != "overall" else "Overall"
            print(f"\n{condition_name}:")
            print(
                f"Number of subjects completing both sessions: {result['n_subjects']}"
            )
            if result["t_stat"] is not None:
                print(f"t = {result['t_stat']:.3f}, p = {result['p_val']:.3f}")
                print(f"Pre-coffee mean accuracy: {result['session1_mean']:.2f}%")
                print(f"Post-coffee mean accuracy: {result['session2_mean']:.2f}%")
            else:
                print("Insufficient sample size for statistical testing")

    # 6. 可视化
    if experiment_config.get("plot", True):
        plot_config = experiment_config.get("plot_config", {})
        fig, ax = plt.subplots(figsize=plot_config.get("figsize", (8, 5)))

        plot_bar_comparison(
            df,
            x_var=plot_config.get("x_var", "condition"),
            y_var=plot_config.get("y_var", "accuracy"),  # 使用accuracy作为y轴变量
            hue_var=plot_config.get("hue_var", "session"),
            title=plot_config.get(
                "title",
                f"{experiment_config['experiment_type']} Task Accuracy Comparison",
            ),
            xlabel=plot_config.get("xlabel", "Condition"),
            ylabel=plot_config.get("ylabel", "Accuracy (%)"),
            palette=plot_config.get("palette", ["lightblue", "lightgreen"]),
            ax=ax,
            show_values=plot_config.get("show_values", True),
        )

        # 添加显著性标记
        x_var = plot_config.get("x_var", "condition")
        y_var = plot_config.get("y_var", "accuracy")  # 使用正确的y变量名
        if x_var == "condition":
            # 遍历各个条件并添加显著性标记
            for i, condition in enumerate(df[x_var].unique()):
                condition_key = str(condition) if condition is not None else "overall"
                if condition_key in ttest_results:
                    result = ttest_results[condition_key]
                    if result["p_val"] is not None and result["n_subjects"] > 1:
                        # 计算显著性标记的高度（使用两个条上限的最大值）
                        condition_data = df[df[x_var] == condition]
                        max_height = (
                            condition_data.groupby("session")[y_var].mean().max() * 1.1
                        )
                        # 添加显著性标记
                        x_positions = [i - 0.2, i + 0.2]  # 根据条形图的宽度和位置调整
                        add_significance_markers(
                            ax, x_positions, result["p_val"], max_height
                        )

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

        if plot_config.get("show", True):
            plt.show()

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
        "extract_function": extract_nback_accuracy,
        "file_patterns": {
            1: "nback_[0-9]*_session1.csv",
            2: "nback_[0-9]*_session2.csv",
        },
        "conditions": [0, 1, 2],  # 不同的n-back水平
        "condition_var": "condition",
        "plot_config": {
            "x_var": "condition",
            "title": "N-back Task Accuracy Comparison",
            "xlabel": "Condition",
            "ylabel": "Accuracy (%)",
            "output_file": "nback_accuracy_comparison.png",
        },
        "output_dir": "output",
    }

    # Stroop实验配置
    stroop_acc_config = {
        "experiment_type": "stroop",
        "extract_function": extract_stroop_accuracy,
        "file_patterns": {
            1: "stroop_stats_[0-9]*_session1.csv",
            2: "stroop_stats_[0-9]*_session2.csv",
        },
        "conditions": ["total"],
        "plot_config": {
            "x_var": "condition",
            "title": "Stroop Task Accuracy Comparison",
            "xlabel": "Session",
            "ylabel": "Accuracy (%)",
            "output_file": "stroop_accuracy_comparison.png",
        },
        "output_dir": "output",
    }

    # BART实验配置
    bart_acc_config = {
        "experiment_type": "bart",
        "extract_function": extract_bart_accuracy,
        "file_patterns": {
            1: "bart_[0-9]*_session1.csv",
            2: "bart_[0-9]*_session2.csv",
        },
        "conditions": ["overall"],  # BART只有一个总体条件
        "plot_config": {
            "x_var": "condition",
            "y_var": "explosion_rate",
            "title": "BART Task Explosion Rate Comparison",
            "xlabel": "Session",
            "ylabel": "Explosion Rate (%)",
            "output_file": "bart_explosion_rate_comparison.png",
        },
        "output_dir": "output",
    }

    # 创建输出目录
    os.makedirs("output", exist_ok=True)

    # 分析实验组数据（不包括8号被试）
    print("==== Analyzing experimental group data (excluding subject 8) ====")
    config_list = [nback_acc_config, stroop_acc_config, bart_acc_config]
    for config in config_list:
        # 更新输出文件名，加上exp前缀
        config["plot_config"]["output_file"] = config["plot_config"][
            "output_file"
        ].replace(".png", "_exp.png")
        config["plot_config"]["title"] = (
            "Experiment Group " + config["plot_config"]["title"]
        )

    exp_results = analyze_combined_accuracy(config_list, exclude_participant=8)

    # 分析对照组数据（只有8号被试）
    print("\n==== Analyzing control group data (only subject 8) ====")
    control_config_list = []
    for config in config_list:
        # 创建新配置，避免修改原始配置
        control_config = config.copy()
        control_config["plot_config"] = config["plot_config"].copy()

        # 更新输出文件名，加上control前缀
        output_file = config["plot_config"]["output_file"]
        control_config["plot_config"]["output_file"] = output_file.replace(
            "_exp.png", "_control.png"
        )
        control_config["plot_config"]["title"] = output_file.replace(
            "Experiment Group", "Control Group"
        )

        control_config_list.append(control_config)

    control_results = analyze_combined_accuracy(control_config_list, only_participant=8)

    print(
        "\nStarting analysis of individual experiment accuracy for experimental group..."
    )
    print("\nAccuracy analysis completed!")
