import pandas as pd
import numpy as np
import re
from glob import glob


def calculate_block_stats(block_df):
    """Calculate statistics for a single block"""
    # Calculate congruent trials (word matches color)
    congruent_trials = block_df[
        block_df.apply(
            lambda x: x["word"]
            == {"Red": "红", "Green": "绿", "Blue": "蓝"}[x["color"]],
            axis=1,
        )
    ]

    # Calculate neutral trials (word is '白')
    neutral_trials = block_df[block_df["word"] == "白"]

    # Calculate incongruent trials (word doesn't match color and is not neutral)
    incongruent_trials = block_df[
        ~block_df.index.isin(congruent_trials.index)
        & ~block_df.index.isin(neutral_trials.index)
    ]

    return {
        "正确率": block_df["correct"].mean() * 100,
        "平均反应时": block_df["rt"].mean() * 1000 + 500,
        "一致条件正确率": congruent_trials["correct"].mean() * 100,
        "不一致条件正确率": incongruent_trials["correct"].mean() * 100,
        "中性词正确率": neutral_trials["correct"].mean() * 100,
        "一致条件反应时": congruent_trials["rt"].mean() * 1000 + 500,
        "不一致条件反应时": incongruent_trials["rt"].mean() * 1000 + 500,
        "中性词反应时": neutral_trials["rt"].mean() * 1000 + 500,
    }


def recalculate_stroop_stats(raw_data_path):
    """
    Recalculate Stroop task statistics from raw data

    Parameters:
    raw_data_path: path to the raw data CSV file
    """
    # Extract participant_id and session from filename
    filename_pattern = r"stroop_(\d+)_session(\d+)\.csv"
    match = re.search(filename_pattern, raw_data_path)
    participant_id = int(match.group(1))
    session = int(match.group(2))

    # Read the raw data
    df = pd.read_csv(raw_data_path)

    # Calculate stats for each block and overall
    stats = []

    # Calculate for each block
    for block in df["block"].unique():
        block_df = df[df["block"] == block]
        block_stats = calculate_block_stats(block_df)
        stats.append(
            {
                "level": "block",
                "block": block + 1,
                **block_stats,
                "participant_id": participant_id,
                "session": session,
            }
        )

    # Calculate overall stats
    total_stats = calculate_block_stats(df)
    stats.append(
        {
            "level": "total",
            "block": "all",
            **total_stats,
            "participant_id": participant_id,
            "session": session,
        }
    )

    # Create and save stats DataFrame
    stats_df = pd.DataFrame(stats)
    output_path = raw_data_path.replace(".csv", "_stats_new.csv")
    stats_df.to_csv(output_path, index=False)

    return stats_df


# 修改循环范围，只处理10号及以后的被试
all_stats = []
for participant in range(10, 34):  # 从10到33号被试
    for session in [1, 2]:
        filename = f"data/stroop_{participant}_session{session}.csv"
        try:
            stats_df = recalculate_stroop_stats(filename)
            all_stats.append(stats_df)
        except FileNotFoundError:
            print(f"File not found: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")


def extract_stroop_detailed_metrics(data_path="data", min_participant_id=10):
    """
    从Stroop实验数据中提取详细的反应时和正确率指标，只处理编号10及以后的被试
    
    Parameters:
    -----------
    data_path : str
        数据文件所在目录
    min_participant_id : int
        最小被试编号（默认为10）
    """
    # 使用更宽松的文件匹配模式
    session1_files = glob(f"{data_path}/stroop_*_session1_stats_new.csv")
    session2_files = glob(f"{data_path}/stroop_*_session2_stats_new.csv")
    
    all_data = []

    # 处理所有文件
    for files, session in [(session1_files, 1), (session2_files, 2)]:
        for file in files:
            try:
                df = pd.read_csv(file)
                
                # 获取被试ID
                participant_id = df['participant_id'].iloc[0]
                
                # 只处理编号10及以后的被试
                if participant_id < min_participant_id:
                    continue

                # 只处理总体数据
                total_data = df[df["level"] == "total"].iloc[0]

                # 添加总体指标
                all_data.append({
                    "participant_id": participant_id,
                    "session": session,
                    "condition": "总体",
                    "accuracy": total_data["正确率"],
                    "rt": total_data["平均反应时"],
                })

                # 添加一致条件指标
                all_data.append({
                    "participant_id": participant_id,
                    "session": session,
                    "condition": "一致",
                    "accuracy": total_data["一致条件正确率"],
                    "rt": total_data["一致条件反应时"],
                })

                # 添加不一致条件指标
                all_data.append({
                    "participant_id": participant_id,
                    "session": session,
                    "condition": "不一致",
                    "accuracy": total_data["不一致条件正确率"],
                    "rt": total_data["不一致条件反应时"],
                })

                # 添加中性条件指标
                all_data.append({
                    "participant_id": participant_id,
                    "session": session,
                    "condition": "中性",
                    "accuracy": total_data["中性词正确率"],
                    "rt": total_data["中性词反应时"],
                })

            except Exception as e:
                print(f"Error processing file {file}: {e}")

    # 创建DataFrame
    df = pd.DataFrame(all_data)
    
    # 打印找到的被试ID
    unique_participants = sorted(df['participant_id'].unique())
    print("\n处理的被试ID:", unique_participants)
    
    return df

# 在调用函数时使用新的参数
df = extract_stroop_detailed_metrics(data_path="data", min_participant_id=10)
