import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from glob import glob

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def analyze_stroop_results():
    """分析Stroop任务的咖啡前后对比"""

    # 读取所有数据文件
    session1_files = glob("data/stroop_stats_*_session1.csv")
    session2_files = glob("data/stroop_stats_*_session2.csv")

    # 存储所有被试的数据
    all_data = []

    # 读取Session 1的数据
    for file in session1_files:
        df = pd.read_csv(file)
        # 只取总体数据（level == 'total'）
        total_data = df[df["level"] == "total"].iloc[0]
        all_data.append(
            {
                "participant_id": total_data["participant_id"],
                "session": 1,
                "accuracy": total_data["正确率"],
                "rt": total_data["平均反应时"],
            }
        )

    # 读取Session 2的数据
    for file in session2_files:
        df = pd.read_csv(file)
        total_data = df[df["level"] == "total"].iloc[0]
        all_data.append(
            {
                "participant_id": total_data["participant_id"],
                "session": 2,
                "accuracy": total_data["正确率"],
                "rt": total_data["平均反应时"],
            }
        )

    # 转换为DataFrame
    results_df = pd.DataFrame(all_data)

    # 计算描述性统计
    desc_stats = results_df.groupby("session").agg(
        {"accuracy": ["mean", "std"], "rt": ["mean", "std"]}
    )

    print("\n描述性统计：")
    print(desc_stats)

    # 找到两个session都有数据的被试
    common_participants = set(
        results_df[results_df["session"] == 1]["participant_id"]
    ) & set(results_df[results_df["session"] == 2]["participant_id"])

    # 准备配对数据
    session1_acc = []
    session2_acc = []
    session1_rt = []
    session2_rt = []

    for pid in common_participants:
        s1_data = results_df[
            (results_df["session"] == 1) & (results_df["participant_id"] == pid)
        ]
        s2_data = results_df[
            (results_df["session"] == 2) & (results_df["participant_id"] == pid)
        ]

        session1_acc.append(s1_data["accuracy"].iloc[0])
        session2_acc.append(s2_data["accuracy"].iloc[0])
        session1_rt.append(s1_data["rt"].iloc[0])
        session2_rt.append(s2_data["rt"].iloc[0])

    # 进行配对t检验
    t_acc, p_acc = stats.ttest_rel(session1_acc, session2_acc)
    t_rt, p_rt = stats.ttest_rel(session1_rt, session2_rt)

    print("\n统计检验结果：")
    print(f"完成两个session的被试数量: {len(common_participants)}")
    print(f"正确率 t检验: t = {t_acc:.3f}, p = {p_acc:.3f}")
    print(f"反应时 t检验: t = {t_rt:.3f}, p = {p_rt:.3f}")

    # 绘制对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 正确率对比图
    sns.barplot(
        x="session",
        y="accuracy",
        data=results_df,
        ax=ax1,
        palette=["lightblue", "lightgreen"],
    )
    ax1.set_title("咖啡前后正确率对比")
    ax1.set_xlabel("实验阶段")
    ax1.set_ylabel("正确率 (%)")
    ax1.set_xticklabels(["咖啡前", "咖啡后"])

    # 反应时对比图
    sns.barplot(
        x="session",
        y="rt",
        data=results_df,
        ax=ax2,
        palette=["lightblue", "lightgreen"],
    )
    ax2.set_title("咖啡前后反应时对比")
    ax2.set_xlabel("实验阶段")
    ax2.set_ylabel("反应时 (ms)")
    ax2.set_xticklabels(["咖啡前", "咖啡后"])

    # 添加误差线
    for ax in [ax1, ax2]:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f")

    plt.tight_layout()
    plt.savefig("output/stroop_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    analyze_stroop_results()
