import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.plot import (
    load_data,
    calculate_descriptive_stats,
    plot_bar_comparison,
    paired_t_test,
    add_significance_markers,
)
import os
import seaborn as sns
from glob import glob
from scipy import stats

# Set font for displaying text
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

# 设置中文字体和图表样式
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")


def extract_nback_rt(session_files):
    """
    从N-back实验数据中提取反应时指标

    Parameters:
    -----------
    session_files : dict
        包含各会话数据文件的字典

    Returns:
    --------
    DataFrame : 包含被试ID、会话、条件和反应时的数据框
    """
    all_data = {
        "task": [],
        "participant_id": [],
        "session": [],
        "condition": [],
        "mean_rt": [],
    }

    for session, files in session_files.items():
        for file in files:
            df = pd.read_csv(file)

            # 只选择正确的试次和目标试次
            correct_trials = df[df["target"] & df["correct"]]

            # 计算每个n-back水平的平均反应时
            for n_back in df["n_back"].unique():
                n_back_trials = correct_trials[correct_trials["n_back"] == n_back]
                if not n_back_trials.empty:
                    # 只选择有效的反应时（不为空且不为0的值）
                    valid_rts = [
                        rt for rt in n_back_trials["rt"] if rt is not None and rt > 0
                    ]
                    if valid_rts:
                        mean_rt = np.mean(valid_rts) * 1000  # 转换为毫秒
                        all_data["task"].append("n-back")
                        all_data["participant_id"].append(df["participant_id"].iloc[0])
                        all_data["session"].append(session)
                        all_data["condition"].append(int(n_back))
                        all_data["mean_rt"].append(mean_rt)

    return pd.DataFrame(all_data)


def extract_stroop_rt(session_files):
    """
    从Stroop实验数据中提取反应时指标

    Parameters:
    -----------
    session_files : dict
        包含各会话数据文件的字典

    Returns:
    --------
    DataFrame : 包含被试ID、会话、条件和反应时的数据框
    """
    all_data = {
        "task": [],
        "participant_id": [],
        "session": [],
        "condition": [],
        "mean_rt": [],
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
            all_data["mean_rt"].append(total_data["平均反应时"])

    return pd.DataFrame(all_data)


def extract_bart_rt(session_files):
    """
    Extract reaction time metrics from BART experiment data

    Parameters:
    -----------
    session_files : dict
        Dictionary containing session data files

    Returns:
    --------
    DataFrame : DataFrame containing participant ID, session, condition and reaction time
    """
    all_data = {
        "task": [],
        "participant_id": [],
        "session": [],
        "condition": [],
        "mean_rt": [],
    }

    for session, files in session_files.items():
        for file in files:
            df = pd.read_csv(file)

            # Filter out non-numeric values and convert to float
            df["rt"] = pd.to_numeric(df["rt"], errors="coerce")

            # Calculate mean reaction time for valid trials
            valid_rts = df[df["rt"].notna() & (df["rt"] > 0)]["rt"]

            if not valid_rts.empty:
                mean_rt = valid_rts.mean() * 1000  # Convert to milliseconds

                all_data["task"].append("BART")
                all_data["participant_id"].append(df["participant_id"].iloc[0])
                all_data["session"].append(session)
                all_data["condition"].append("overall")
                all_data["mean_rt"].append(mean_rt)

    return pd.DataFrame(all_data)


def extract_emotion_rt(session_files):
    """
    Extract reaction time metrics from emotion regulation experiment data

    Parameters:
    -----------
    session_files : dict
        Dictionary containing session data files

    Returns:
    --------
    DataFrame : DataFrame containing participant ID, session, condition and reaction time
    """
    all_data = {
        "task": [],
        "participant_id": [],
        "session": [],
        "condition": [],
        "mean_rt": [],
    }

    for session, files in session_files.items():
        for file in files:
            df = pd.read_csv(file)

            # Process each phase separately (baseline and regulation)
            for phase in df["phase"].unique():
                phase_data = df[df["phase"] == phase]

                # Calculate mean reaction time using rating_rt (as specified)
                valid_rts = phase_data[
                    phase_data["rating_rt"].notna() & (phase_data["rating_rt"] > 0)
                ]["rating_rt"]

                if not valid_rts.empty:
                    mean_rt = (
                        valid_rts.mean() * 1000
                    )  # Convert to milliseconds if needed

                    all_data["task"].append("emotion")
                    all_data["participant_id"].append(df["participant_id"].iloc[0])
                    all_data["session"].append(session)
                    all_data["condition"].append(phase)
                    all_data["mean_rt"].append(mean_rt)

    return pd.DataFrame(all_data)


def extract_stroop_detailed_metrics(data_path="data"):
    """
    从Stroop实验数据中提取详细的反应时和正确率指标
    
    Parameters:
    -----------
    data_path : str
        数据文件所在目录
        
    Returns:
    --------
    DataFrame : 包含被试ID、会话、条件、反应时和正确率的数据框
    """
    # 使用glob获取所有stroop统计文件
    session1_files = glob(f"{data_path}/stroop_stats_*_session1.csv")
    session2_files = glob(f"{data_path}/stroop_stats_*_session2.csv")
    
    all_data = []
    
    # 处理所有文件
    for files, session in [(session1_files, 1), (session2_files, 2)]:
        for file in files:
            try:
                df = pd.read_csv(file)
                
                # 只处理总体数据
                total_data = df[df["level"] == "total"].iloc[0]
                
                # 提取被试ID
                participant_id = total_data["participant_id"]
                
                # 添加总体指标
                all_data.append({
                    "participant_id": participant_id,
                    "session": session,
                    "condition": "总体",
                    "accuracy": total_data["正确率"],
                    "rt": total_data["平均反应时"]
                })
                
                # 添加一致条件指标
                all_data.append({
                    "participant_id": participant_id,
                    "session": session,
                    "condition": "一致",
                    "accuracy": total_data["一致条件正确率"],
                    "rt": total_data["一致条件反应时"]
                })
                
                # 添加不一致条件指标
                all_data.append({
                    "participant_id": participant_id,
                    "session": session,
                    "condition": "不一致",
                    "accuracy": total_data["不一致条件正确率"],
                    "rt": total_data["不一致条件反应时"]
                })
                
                # 添加中性词条件指标
                all_data.append({
                    "participant_id": participant_id,
                    "session": session,
                    "condition": "中性",
                    "accuracy": total_data["中性词正确率"],
                    "rt": total_data["中性词反应时"]
                })
                
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")
    
    return pd.DataFrame(all_data)


def analyze_experiment_rt(config, exclude_participant=None, only_participant=None):
    """
    通用实验反应时分析主函数

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
        config["experiment_type"],
        config.get("data_path", "data"),
        config.get("file_patterns", None),
    )

    # 2. 提取数据
    extract_func = config["extract_function"]
    df = extract_func(session_files)

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

    # If there's no data after filtering, skip this experiment
    if df.empty:
        print(
            f"Experiment {config['experiment_type']} has no valid data after filtering"
        )

    # 3. 描述性统计
    group_vars = config.get("group_vars", ["session", "condition"])
    desc_stats = calculate_descriptive_stats(df, group_vars)
    results["descriptive_stats"] = desc_stats

    # 4. 统计检验
    ttest_results = {}
    conditions = config.get("conditions", [None])
    condition_var = config.get("condition_var", "condition")

    for condition in conditions:
        condition_key = str(condition) if condition is not None else "overall"
        ttest_results[condition_key] = paired_t_test(df, condition_var, condition)

    results["ttest_results"] = ttest_results

    # 5. 打印结果
    print(f"\n{config['experiment_type'].upper()} Task Analysis Results:")
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
                print(
                    f"Pre-coffee mean reaction time: {result['session1_mean']:.2f} ms"
                )
                print(
                    f"Post-coffee mean reaction time: {result['session2_mean']:.2f} ms"
                )
            else:
                print("Insufficient sample size for statistical testing")

    # 6. 可视化
    if config.get("plot", True):
        plot_config = config.get("plot_config", {})
        fig, ax = plt.subplots(figsize=plot_config.get("figsize", (8, 5)))

        plot_bar_comparison(
            df,
            x_var=plot_config.get("x_var", "condition"),
            y_var=plot_config.get("y_var", "mean_rt"),
            hue_var=plot_config.get("hue_var", "session"),
            title=plot_config.get(
                "title",
                f"{config['experiment_type']} Task Reaction Time Comparison",
            ),
            xlabel=plot_config.get("xlabel", "Condition"),
            ylabel=plot_config.get("ylabel", "Reaction Time (ms)"),
            palette=plot_config.get("palette", ["lightblue", "lightgreen"]),
            ax=ax,
            show_values=plot_config.get("show_values", True),
        )

        # 添加显著性标记
        x_var = plot_config.get("x_var", "condition")
        y_var = plot_config.get("y_var", "mean_rt")
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
        output_dir = config.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            plot_config.get(
                "output_file",
                f"{config['experiment_type']}_rt_comparison.png",
            ),
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

        if plot_config.get("show", True):
            plt.show()

    return results


def analyze_combined_experiments(
    experiments_config, exclude_participant=None, only_participant=None
):
    """
    分析多个实验并进行对比

    Parameters:
    -----------
    experiments_config : list
        实验配置列表，每个元素为一个实验配置字典
    exclude_participant : int or lis-t
        要排除的被试ID，如果为None则不排除任何被试
    only_participant : int or list
        只分析特定被试ID，如果为None则分析所有被试
    """
    # 处理所有实验
    all_results = {}

    for config in experiments_config:
        # 4. 分析实验
        results = analyze_experiment_rt(config, exclude_participant, only_participant)
        all_results[config["experiment_type"]] = results

    return all_results


def analyze_stroop_detailed():
    """分析Stroop任务在不同条件下的反应时和正确率"""
    
    # 提取数据
    df = extract_stroop_detailed_metrics()
    
    if df.empty:
        print("未找到有效的Stroop任务数据")
        return
    
    # 创建输出目录
    os.makedirs("output", exist_ok=True)
    
    # 计算描述性统计
    desc_stats = df.groupby(["session", "condition"]).agg({
        "accuracy": ["mean", "std", "count"],
        "rt": ["mean", "std", "count"]
    }).round(2)
    
    print("\n描述性统计：")
    print(desc_stats)
    
    # 找到两个session都有数据的被试
    common_participants = set(df[df["session"] == 1]["participant_id"]) & set(df[df["session"] == 2]["participant_id"])
    print(f"\n完成两个session的被试数量: {len(common_participants)}")
    
    # 进行配对t检验（按条件分组）
    ttest_results = {}
    
    for condition in df["condition"].unique():
        condition_data = df[df["condition"] == condition]
        
        # 准备配对数据
        paired_data = []
        
        for pid in common_participants:
            s1_data = condition_data[(condition_data["session"] == 1) & (condition_data["participant_id"] == pid)]
            s2_data = condition_data[(condition_data["session"] == 2) & (condition_data["participant_id"] == pid)]
            
            if not s1_data.empty and not s2_data.empty:
                paired_data.append({
                    "participant_id": pid,
                    "session1_acc": s1_data["accuracy"].iloc[0],
                    "session2_acc": s2_data["accuracy"].iloc[0],
                    "session1_rt": s1_data["rt"].iloc[0],
                    "session2_rt": s2_data["rt"].iloc[0]
                })
        
        if paired_data:
            paired_df = pd.DataFrame(paired_data)
            
            # 正确率t检验
            t_acc, p_acc = stats.ttest_rel(paired_df["session1_acc"], paired_df["session2_acc"])
            
            # 反应时t检验
            t_rt, p_rt = stats.ttest_rel(paired_df["session1_rt"], paired_df["session2_rt"])
            
            ttest_results[condition] = {
                "n_subjects": len(paired_df),
                "accuracy": {
                    "t_stat": t_acc,
                    "p_val": p_acc,
                    "session1_mean": paired_df["session1_acc"].mean(),
                    "session2_mean": paired_df["session2_acc"].mean()
                },
                "rt": {
                    "t_stat": t_rt,
                    "p_val": p_rt,
                    "session1_mean": paired_df["session1_rt"].mean(),
                    "session2_mean": paired_df["session2_rt"].mean()
                }
            }
    
    # 打印统计检验结果
    print("\n统计检验结果：")
    for condition, result in ttest_results.items():
        print(f"\n{condition}条件:")
        print(f"完成两个session的被试数量: {result['n_subjects']}")
        
        print("正确率对比:")
        print(f"t = {result['accuracy']['t_stat']:.3f}, p = {result['accuracy']['p_val']:.3f}")
        print(f"咖啡前平均正确率: {result['accuracy']['session1_mean']:.2f}%")
        print(f"咖啡后平均正确率: {result['accuracy']['session2_mean']:.2f}%")
        
        print("反应时对比:")
        print(f"t = {result['rt']['t_stat']:.3f}, p = {result['rt']['p_val']:.3f}")
        print(f"咖啡前平均反应时: {result['rt']['session1_mean']:.2f} ms")
        print(f"咖啡后平均反应时: {result['rt']['session2_mean']:.2f} ms")
    
    # 绘制反应时对比图
    plt.figure(figsize=(12, 6))
    
    # 准备绘图数据
    plot_data = df.copy()
    
    # 将中文条件名转换为英文
    condition_mapping = {
        "总体": "Overall",
        "一致": "Congruent",
        "不一致": "Incongruent",
        "中性": "Neutral"
    }
    plot_data["condition"] = plot_data["condition"].map(condition_mapping)
    
    # 将中文session名转换为英文
    plot_data["session"] = plot_data["session"].map({1: "Pre-coffee", 2: "Post-coffee"})
    
    # 绘制反应时条形图
    ax = sns.barplot(x="condition", y="rt", hue="session", data=plot_data, 
                    palette=["lightblue", "lightgreen"], errorbar="se")
    
    # 添加标题和标签
    plt.title("Stroop Task Reaction Time Comparison by Condition", fontsize=14)
    plt.xlabel("Condition", fontsize=12)
    plt.ylabel("Reaction Time (ms)", fontsize=12)
    plt.legend(title="Experiment Stage")
    
    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=9)
    
    # 添加显著性标记
    for i, condition in enumerate(plot_data["condition"].unique()):
        orig_condition = [k for k, v in condition_mapping.items() if v == condition][0]
        if orig_condition in ttest_results:
            result = ttest_results[orig_condition]["rt"]
            p_val = result["p_val"]
            
            # 计算显著性标记的高度
            max_height = max(result["session1_mean"], result["session2_mean"]) * 1.1
            
            # 添加显著性标记
            if p_val < 0.001:
                plt.text(i, max_height, "***", ha='center', fontsize=12)
            elif p_val < 0.01:
                plt.text(i, max_height, "**", ha='center', fontsize=12)
            elif p_val < 0.05:
                plt.text(i, max_height, "*", ha='center', fontsize=12)
            else:
                plt.text(i, max_height, "ns", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("output/stroop_rt_conditions_comparison.png", dpi=300, bbox_inches="tight")
    
    # 绘制正确率对比图
    plt.figure(figsize=(12, 6))
    
    # 绘制正确率条形图
    ax = sns.barplot(x="condition", y="accuracy", hue="session", data=plot_data, 
                    palette=["lightblue", "lightgreen"], errorbar="se")
    
    # 添加标题和标签
    plt.title("Stroop Task Accuracy Comparison by Condition", fontsize=14)
    plt.xlabel("Condition", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.legend(title="Experiment Stage")
    
    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=9)
    
    # 添加显著性标记
    for i, condition in enumerate(plot_data["condition"].unique()):
        orig_condition = [k for k, v in condition_mapping.items() if v == condition][0]
        if orig_condition in ttest_results:
            result = ttest_results[orig_condition]["accuracy"]
            p_val = result["p_val"]
            
            # 计算显著性标记的高度
            max_height = max(result["session1_mean"], result["session2_mean"]) * 1.05
            
            # 添加显著性标记
            if p_val < 0.001:
                plt.text(i, max_height, "***", ha='center', fontsize=12)
            elif p_val < 0.01:
                plt.text(i, max_height, "**", ha='center', fontsize=12)
            elif p_val < 0.05:
                plt.text(i, max_height, "*", ha='center', fontsize=12)
            else:
                plt.text(i, max_height, "ns", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("output/stroop_accuracy_conditions_comparison.png", dpi=300, bbox_inches="tight")
    
    # 显示图表
    plt.show()


# 示例用法
if __name__ == "__main__":
    print("Starting analysis of experimental data...")

    # 创建输出目录
    os.makedirs("output", exist_ok=True)

    # N-back实验配置
    nback_config = {
        "experiment_type": "nback",
        "extract_function": extract_nback_rt,
        "file_patterns": {
            1: "nback_[0-9]*_session1.csv",
            2: "nback_[0-9]*_session2.csv",
        },
        "conditions": [0, 1, 2],  # 不同的n-back水平
        "condition_var": "condition",
        "plot_config": {
            "x_var": "condition",
            "title": "N-back task reaction time comparison",
            "xlabel": "N-back level",
            "output_file": "nback_rt_comparison.png",
        },
        "output_dir": "output",
    }

    # Stroop实验配置
    stroop_config = {
        "experiment_type": "stroop",
        "extract_function": extract_stroop_rt,
        "file_patterns": {
            1: "stroop_stats_[0-9]*_session1.csv",
            2: "stroop_stats_[0-9]*_session2.csv",
        },
        "conditions": ["total"],  # Stroop只有一个总体条件
        "plot_config": {
            "x_var": "condition",
            "title": "Stroop task reaction time comparison",
            "xlabel": "Experiment stage",
            "output_file": "stroop_rt_comparison.png",
        },
        "output_dir": "output",
    }

    # BART实验配置
    bart_config = {
        "experiment_type": "bart",
        "extract_function": extract_bart_rt,
        "file_patterns": {
            1: "bart_[0-9]*_session1.csv",
            2: "bart_[0-9]*_session2.csv",
        },
        "conditions": ["overall"],  # BART只分析总体反应时
        "plot_config": {
            "x_var": "condition",
            "title": "BART task reaction time comparison",
            "xlabel": "Experiment stage",
            "output_file": "bart_rt_comparison.png",
        },
        "output_dir": "output",
    }

    # # 情绪实验配置
    # emotion_config = {
    #     "experiment_type": "emotion",
    #     "extract_function": extract_emotion_rt,
    #     "file_patterns": {
    #         1: "emotion_[0-9]*_session1.csv",
    #         2: "emotion_[0-9]*_session2.csv",
    #     },
    #     "conditions": ["baseline", "regulation"],  # 情绪实验有基线和调节两个阶段
    #     "condition_var": "condition",
    #     "plot_config": {
    #         "x_var": "condition",
    #         "title": "Emotion task reaction time comparison",
    #         "xlabel": "Experiment stage",
    #         "ylabel": "Rating reaction time (ms)",
    #         "output_file": "emotion_rt_comparison.png",
    #     },
    #     "output_dir": "output",
    # }

    # 所有实验配置
    config_list = [nback_config, stroop_config, bart_config]

    # 分析实验组数据（除8号被试外）
    print("==== Analyzing experimental group data (excluding subject 8) ====")
    for config in config_list:
        # 更新输出文件名，加上exp前缀
        config["plot_config"]["output_file"] = config["plot_config"][
            "output_file"
        ].replace(".png", "_exp.png")
        config["plot_config"]["title"] = (
            "Experimental group " + config["plot_config"]["title"]
        )

    print("Starting analysis of individual experiments for experimental group...")
    exp_results = analyze_combined_experiments(config_list, exclude_participant=[0,1,2,3,4,5,6,7,8])

    # # 分析对照组数据（只有8号被试）
    # print("\n==== Analyzing control group data (only subject 8) ====")
    # control_config_list = []
    # for config in config_list:
    #     # 创建新配置，避免修改原始配置
    #     control_config = config.copy()
    #     control_config["plot_config"] = config["plot_config"].copy()

    #     # 更新输出文件名，加上control前缀
    #     output_file = config["plot_config"]["output_file"]
    #     control_config["plot_config"]["output_file"] = output_file.replace(
    #         "_exp.png", "_control.png"
    #     )
    #     control_config["plot_config"]["title"] = control_config["plot_config"][
    #         "title"
    #     ].replace("Experimental group", "Control group")

    #     control_config_list.append(control_config)

    # print("Starting analysis of individual experiments for control group...")
    # control_results = analyze_combined_experiments(
    #     control_config_list, only_participant=8
    # )

    print("Analysis completed!")

    # 分析Stroop任务在不同条件下的反应时和正确率
    analyze_stroop_detailed()
