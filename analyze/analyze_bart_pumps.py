import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
from scipy import stats
from utils.plot import add_significance_markers


def load_bart_data(data_path="data"):
    """
    Load BART experiment data files
    
    Parameters:
    -----------
    data_path : str
        Path to data directory
        
    Returns:
    --------
    dict : Dictionary containing session data files
    """
    session_files = {
        1: [os.path.join(data_path, f"bart_{i}_session1.csv") for i in range(11, 32)],
        2: [os.path.join(data_path, f"bart_{i}_session2.csv") for i in range(11, 32)],
    }
    
    print(f"Found {len(session_files[1])} files for session 1")
    print(f"Found {len(session_files[2])} files for session 2")
    
    return session_files


def extract_bart_pumps(session_files):
    """
    Extract pump data from BART experiment
    
    Parameters:
    -----------
    session_files : dict
        Dictionary containing session data files
        
    Returns:
    --------
    DataFrame : DataFrame containing participant ID, session, and pump data
    """
    all_data = {
        "participant_id": [],
        "session": [],
        "mean_pumps": [],
        "mean_pumps_unexploded": [],
        "explosion_rate": []
    }
    
    for session, files in session_files.items():
        for file in files:
            df = pd.read_csv(file)
            df = df[df["pumps"].notna()]
            
            # Extract participant ID from filename
            file_name = os.path.basename(file)
            participant_id = int(file_name.split("_")[1])
            
            # Calculate pump metrics
            mean_pumps = df["pumps"].mean()
            
            # Calculate mean pumps for unexploded balloons
            unexploded_trials = df[df["exploded"] == False]
            mean_pumps_unexploded = 0
            if len(unexploded_trials) > 0:
                mean_pumps_unexploded = unexploded_trials["pumps"].mean()
            
            # Calculate explosion rate
            explosion_rate = df["exploded"].mean() * 100
            
            # Add data to results
            all_data["participant_id"].append(participant_id)
            all_data["session"].append(session)
            all_data["mean_pumps"].append(mean_pumps)
            all_data["mean_pumps_unexploded"].append(mean_pumps_unexploded)
            all_data["explosion_rate"].append(explosion_rate)
            
            print(f"Processed BART pumps for participant {participant_id}, session {session}")
    
    return pd.DataFrame(all_data)


def paired_t_test(df, measure):
    """
    Perform paired t-test on pump data
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing pump data
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


def plot_bart_pumps(df, measure, title, ylabel, output_file=None):
    """
    Plot BART pumps comparison between sessions
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing pump data
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
    
    # 使用更好的方式创建条形图
    sns.set_style("whitegrid")
    
    # 计算每个会话的平均值和标准误差
    summary = df.groupby('session')[measure].agg(['mean', 'std', 'count'])
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    
    # 创建条形图
    x = np.array([0.25, 0.75])  # 调整条形位置，使它们更靠近
    width = 0.4
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, summary['mean'], width, 
                 yerr=summary['se'], 
                 color=['lightblue', 'lightgreen'],
                 capsize=5, edgecolor='black', linewidth=1)
    
    # 设置标签和标题
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Experiment Phase", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # 移除x轴刻度，添加自定义标签
    ax.set_xticks([])
    ax.set_xlim(-0.2, 1.2)
    
    # 添加图例
    ax.legend(bars, ["Before Coffee", "After Coffee"], loc='lower right')
    
    # 执行t检验
    results = paired_t_test(df, measure)
    
    # 添加显著性标记（如果适用）
    if results["p_val"] is not None and results["n_subjects"] > 1:
        max_height = max(summary['mean']) + max(summary['se']) * 2
        
        # 添加显著性线和星号
        if results["p_val"] < 0.001:
            sig_symbol = "***"
        elif results["p_val"] < 0.01:
            sig_symbol = "**"
        elif results["p_val"] < 0.05:
            sig_symbol = "*"
        else:
            sig_symbol = "ns"
            
        if results["p_val"] < 0.05:  # 只在显著时添加线
            ax.plot([x[0], x[1]], [max_height*1.05, max_height*1.05], 'k-', linewidth=1.5)
            ax.text((x[0]+x[1])/2, max_height*1.1, sig_symbol, ha='center', va='bottom', fontsize=14)
    
    plt.tight_layout()
    
    # 如果提供了输出文件，则保存图形
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    
    plt.show()


def analyze_bart_pumps():
    """
    Analyze BART pump data and compare between sessions
    """
    print("Starting analysis of BART task pump data...")
    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    session_files = load_bart_data()
    
    # Extract pump data
    pumps_df = extract_bart_pumps(session_files)
    
    if pumps_df.empty:
        print("No valid BART data found")
        return
    
    # 将数据分为实验组（除8号外）和对照组（8号）
    exp_df = pumps_df[pumps_df["participant_id"] > 8]
    
    print("\n==== Analyzing experimental group data (excluding subject 8) ====")
    analyze_group_pumps(exp_df, output_dir, "exp")


def analyze_group_pumps(df, output_dir, group_name):
    """
    Analyze BART pumps for a specific group
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing pump data for the group
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
    measures = ["mean_pumps", "mean_pumps_unexploded", "explosion_rate"]
    
    print("\nStatistical Test Results:")
    for measure in measures:
        results = paired_t_test(df, measure)
        measure_name = {
            "mean_pumps": "Mean Pumps",
            "mean_pumps_unexploded": "Mean Pumps for Unexploded Balloons",
            "explosion_rate": "Explosion Rate (%)"
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
            "title": f"{group_name.upper()} Group BART Task Mean Pumps",
            "ylabel": "Mean Pumps",
            "measure": "mean_pumps",
            "output_file": f"bart_{group_name}_mean_pumps.png",
        },
        {
            "title": f"{group_name.upper()} Group BART Task Explosion Rate",
            "ylabel": "Explosion Rate (%)",
            "measure": "explosion_rate",
            "output_file": f"bart_{group_name}_explosion_rate.png",
        },
    ]
    
    for config in plot_configs:
        plot_bart_pumps(
            df,
            config["measure"],
            config["title"],
            config["ylabel"],
            os.path.join(output_dir, config["output_file"]),
        )


if __name__ == "__main__":
    analyze_bart_pumps() 