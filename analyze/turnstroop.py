import pandas as pd
import numpy as np
import re


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
        "平均反应时": block_df["rt"].mean() * 1000,
        "一致条件正确率": congruent_trials["correct"].mean() * 100,
        "不一致条件正确率": incongruent_trials["correct"].mean() * 100,
        "中性词正确率": neutral_trials["correct"].mean() * 100,
        "一致条件反应时": congruent_trials["rt"].mean() * 1000,
        "不一致条件反应时": incongruent_trials["rt"].mean() * 1000,
        "中性词反应时": neutral_trials["rt"].mean() * 1000,
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


# 处理数据
# Process all stroop data files from participant 0 to 27, sessions 1 and 2
all_stats = []
for participant in range(3, 31):  # 0 to 30
    for session in [1, 2]:
        filename = f"data/stroop_{participant}_session{session}.csv"
        stats_df = recalculate_stroop_stats(filename)
        all_stats.append(stats_df)
