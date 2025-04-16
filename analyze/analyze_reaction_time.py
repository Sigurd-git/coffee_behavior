import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.plot import (
    calculate_descriptive_stats,
    plot_bar_comparison,
    paired_t_test,
    add_significance_markers,
)
from utils.tasks import (
    extract_nback,
    extract_stroop,
    extract_bart,
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
        "rt": [],
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
                    rt = valid_rts.mean() * 1000  # Convert to milliseconds if needed

                    all_data["task"].append("emotion")
                    all_data["participant_id"].append(df["participant_id"].iloc[0])
                    all_data["session"].append(session)
                    all_data["condition"].append(phase)
                    all_data["rt"].append(rt)

    return pd.DataFrame(all_data)


def extract_stroop_detailed_metrics(data_path="data", exclude_participant=None):
    """
    从Stroop实验数据中提取详细的反应时和正确率指标
    """
    # 使用更宽松的文件匹配模式
    # 排除102、106、109号被试
    session1_files = [f for f in glob(f"{data_path}/stroop_*_session1_stats_new.csv") 
                     if not any(str(pid) in f for pid in [13, 19, 25, 102, 106, 109])]
    session2_files = [f for f in glob(f"{data_path}/stroop_*_session2_stats_new.csv")
                     if not any(str(pid) in f for pid in [13, 19, 25, 102, 106, 109])]
    print("\nFound files:")
    print("Session 1 files:", session1_files)
    print("Session 2 files:", session2_files)
    
    all_data = []

    # 处理所有文件
    for files, session in [(session1_files, 1), (session2_files, 2)]:
        for file in files:
            try:
                df = pd.read_csv(file)
                
                # 打印当前处理的文件和被试ID
                participant_id = df['participant_id'].iloc[0]
                print(f"\nProcessing file: {file}")
                print(f"Participant ID: {participant_id}")

                if exclude_participant is not None and participant_id in exclude_participant:
                    print(f"Skipping excluded participant {participant_id}")
                    continue

                # 只处理总体数据
                total_data = df[df["level"] == "total"].iloc[0]

                # Add overall metrics
                all_data.append(
                    {
                        "participant_id": participant_id,
                        "session": session,
                        "condition": "总体",
                        "accuracy": total_data["正确率"],
                        "rt": total_data["平均反应时"],
                    }
                )

                # Add congruent condition metrics
                all_data.append(
                    {
                        "participant_id": participant_id,
                        "session": session,
                        "condition": "一致",
                        "accuracy": total_data["一致条件正确率"],
                        "rt": total_data["一致条件反应时"],
                    }
                )

                # Add incongruent condition metrics
                all_data.append(
                    {
                        "participant_id": participant_id,
                        "session": session,
                        "condition": "不一致",
                        "accuracy": total_data["不一致条件正确率"],
                        "rt": total_data["不一致条件反应时"],
                    }
                )

                # Add neutral condition metrics
                all_data.append(
                    {
                        "participant_id": participant_id,
                        "session": session,
                        "condition": "中性",
                        "accuracy": total_data["中性词正确率"],
                        "rt": total_data["中性词反应时"],
                    }
                )

                # Add incongruent-neutral difference
                all_data.append(
                    {
                        "participant_id": participant_id,
                        "session": session,
                        "condition": "不一致-中性",
                        "accuracy": total_data["不一致条件正确率"]
                        - total_data["中性词正确率"],
                        "rt": total_data["不一致条件反应时"]
                        - total_data["中性词反应时"],
                    }
                )

                # Add congruent-neutral difference
                all_data.append(
                    {
                        "participant_id": participant_id,
                        "session": session,
                        "condition": "一致-中性",
                        "accuracy": total_data["一致条件正确率"]
                        - total_data["中性词正确率"],
                        "rt": total_data["一致条件反应时"] - total_data["中性词反应时"],
                    }
                )

            except Exception as e:
                print(f"Error processing file {file}: {e}")
                
    # 打印找到的所有被试ID
    unique_participants = sorted(list(set(df['participant_id'] for df in all_data)))
    print("\nFound participant IDs:", unique_participants)
    
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

    # 2. 提取数据
    extract_func = config["extract_function"]
    df = extract_func(config)

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
            y_var=plot_config.get("y_var", "rt"),
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
        y_var = plot_config.get("y_var", "rt")
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
    """分析Stroop任务在不同条件下的反应时、正确率和IES"""

    # 提取数据
    df = extract_stroop_detailed_metrics(exclude_participant=[0, 1, 8, 30])

    if df.empty:
        print("未找到有效的Stroop任务数据")
        return

    # 创建输出目录
    os.makedirs("output", exist_ok=True)

    # 计算描述性统计
    desc_stats = (
        df.groupby(["session", "condition"])
        .agg({"accuracy": ["mean", "std", "count"], "rt": ["mean", "std", "count"]})
        .round(2)
    )

    print("\n描述性统计：")
    print(desc_stats)

    # 找到两个session都有数据的被试
    common_participants = set(df[df["session"] == 1]["participant_id"]) & set(
        df[df["session"] == 2]["participant_id"]
    )
    print(f"\n完成两个session的被试数量: {len(common_participants)}")

    # 创建结果DataFrame
    results_data = []
    
    for pid in common_participants:
        for session in [1, 2]:
            session_data = df[df['session'] == session]
            subject_data = session_data[session_data['participant_id'] == pid]
            
            # 获取各条件的反应时和正确率
            incongruent_data = subject_data[subject_data['condition'] == '不一致'].iloc[0]
            congruent_data = subject_data[subject_data['condition'] == '一致'].iloc[0]
            neutral_data = subject_data[subject_data['condition'] == '中性'].iloc[0]
            
            # 计算各条件的IES (RT/accuracy)
            incongruent_ies = incongruent_data['rt'] / (incongruent_data['accuracy'] / 100)
            congruent_ies = congruent_data['rt'] / (congruent_data['accuracy'] / 100)
            neutral_ies = neutral_data['rt'] / (neutral_data['accuracy'] / 100)
            
            # 计算效应值
            incongruent_congruent = incongruent_data['rt'] - congruent_data['rt']
            incongruent_neutral = incongruent_data['rt'] - neutral_data['rt']
            neutral_congruent = neutral_data['rt'] - congruent_data['rt']
            
            # 计算IES效应值
            incongruent_congruent_ies = incongruent_ies - congruent_ies
            incongruent_neutral_ies = incongruent_ies - neutral_ies
            neutral_congruent_ies = neutral_ies - congruent_ies
            
            results_data.append({
                'Subject_ID': pid,
                'Session': f'Session {session}',
                'Incongruent-Congruent_RT': round(incongruent_congruent, 2),
                'Incongruent-Neutral_RT': round(incongruent_neutral, 2),
                'Neutral-Congruent_RT': round(neutral_congruent, 2),
                'Incongruent-Congruent_IES': round(incongruent_congruent_ies, 2),
                'Incongruent-Neutral_IES': round(incongruent_neutral_ies, 2),
                'Neutral-Congruent_IES': round(neutral_congruent_ies, 2)
            })
    
    # 创建DataFrame并保存为Excel
    results_df = pd.DataFrame(results_data)
    
    # 重新组织数据以使session1和session2并排显示
    pivot_df = results_df.pivot(index='Subject_ID', 
                              columns='Session', 
                              values=['Incongruent-Congruent_RT', 
                                    'Incongruent-Neutral_RT', 
                                    'Neutral-Congruent_RT',
                                    'Incongruent-Congruent_IES',
                                    'Incongruent-Neutral_IES',
                                    'Neutral-Congruent_IES'])
    
    # 计算描述性统计
    desc_stats = pivot_df.agg(['mean', 'std']).round(2)
    
    # 将描述性统计添加到主DataFrame
    final_df = pd.concat([pivot_df, desc_stats])
    
    # 保存到Excel
    excel_path = 'output/stroop_interference_effects.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 保存主要结果
        final_df.to_excel(writer, sheet_name='Detailed Results')
        
        # 创建配对t检验结果的sheet
        t_test_results = []
        measures = ['Incongruent-Congruent_RT', 'Incongruent-Neutral_RT', 'Neutral-Congruent_RT',
                   'Incongruent-Congruent_IES', 'Incongruent-Neutral_IES', 'Neutral-Congruent_IES']
        
        for measure in measures:
            session1_data = results_df[results_df['Session'] == 'Session 1'][measure]
            session2_data = results_df[results_df['Session'] == 'Session 2'][measure]
            t_stat, p_val = stats.ttest_rel(session1_data, session2_data)
            
            t_test_results.append({
                'Measure': measure,
                't_statistic': round(t_stat, 3),
                'p_value': round(p_val, 3),
                'Session1_Mean': round(session1_data.mean(), 2),
                'Session1_SD': round(session1_data.std(), 2),
                'Session2_Mean': round(session2_data.mean(), 2),
                'Session2_SD': round(session2_data.std(), 2)
            })
        
        # 保存t检验结果
        pd.DataFrame(t_test_results).to_excel(writer, sheet_name='T-test Results', index=False)
    
    print(f"\nResults have been saved to: {excel_path}")
    
    # 打印结果预览
    print("\nDetailed Results Preview:")
    print(pivot_df.round(2))
    
    print("\nDescriptive Statistics:")
    print(desc_stats)
    
    print("\nPaired t-test Results:")
    for result in t_test_results:
        print(f"\n{result['Measure']}:")
        print(f"t = {result['t_statistic']}, p = {result['p_value']}")
        print(f"Session 1: M = {result['Session1_Mean']}, SD = {result['Session1_SD']}")
        print(f"Session 2: M = {result['Session2_Mean']}, SD = {result['Session2_SD']}")


# 示例用法
if __name__ == "__main__":
    print("Starting analysis of experimental data...")

    # 创建输出目录
    os.makedirs("output", exist_ok=True)

    # N-back实验配置
    nback_config = {
        "experiment_type": "nback",
        "extract_function": extract_nback,
        "conditions": ["0-back", "1-back", "2-back"],  # 不同的n-back水平
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
        "extract_function": extract_stroop,
        "conditions": [
            "Consistent",
            "Inconsistent",
            "Neutral",
        ],
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
        "extract_function": extract_bart,
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
    exp_results = analyze_combined_experiments(
        config_list, exclude_participant=[0, 1, 8]
    )

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
    # analyze_stroop_detailed()
