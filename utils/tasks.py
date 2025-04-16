import os
import pandas as pd
from utils.plot import load_data
import numpy as np


def split_emo_normal(result_df):
    emo_df = result_df[result_df["participant_id"] > 100]
    normal_df = result_df[result_df["participant_id"] < 100]
    emo_df["group"] = "emo"
    normal_df["group"] = "normal"
    result_df = pd.concat([emo_df, normal_df])
    return result_df


# 提取n-back任务的正确率和rt
def extract_nback(experiment_config):
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
    session_files = load_data(
        experiment_config["experiment_type"],
        experiment_config.get("data_path", "data"),
    )

    all_files_data = []
    for session, files in session_files.items():
        # Collect all data first

        for file in files:
            df = pd.read_csv(file)
            all_files_data.append(df)

    combined_data = pd.concat(all_files_data, ignore_index=True)

    # 只选择正确的试次和目标试次
    target_trials = combined_data[combined_data["target"]]

    # Group by session, participant_id, and n_back level to calculate accuracy
    result_df = (
        target_trials.groupby(["session", "participant_id", "n_back"])
        .agg(
            total_trials=("target", "count"),
            correct_trials=("correct", "sum"),
            rt=("rt", "mean"),
        )
        .reset_index()
    )

    # Calculate accuracy as percentage
    result_df["accuracy"] = (
        result_df["correct_trials"] / result_df["total_trials"]
    ) * 100
    result_df["rt"] = result_df["rt"] * 1000
    result_df["task"] = "n-back"
    result_df["condition"] = result_df["n_back"].astype(str) + "-back"
    result_df = split_emo_normal(result_df)
    return result_df


# 提取Stroop任务的正确率和rt
def extract_stroop(experiment_config):
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
    session_files = load_data(
        experiment_config["experiment_type"],
        experiment_config.get("data_path", "data"),
    )

    color_map = {"Red": "红", "Green": "绿", "Blue": "蓝"}
    all_files_data = []
    for session, files in session_files.items():
        # Collect all data first
        for file in files:
            df = pd.read_csv(file)
            subject = int(file.split("_")[1])
            # Add session information to the dataframe
            df["session"] = session
            df["participant_id"] = subject
            all_files_data.append(df)

    # Combine all data
    combined_data = pd.concat(all_files_data, ignore_index=True)
    combined_data["color_cn"] = combined_data["color"].map(color_map)

    def get_condition(row):
        if row["word"] == "白":
            return "Neutral"
        elif row["word"] == row["color_cn"]:
            return "Consistent"
        else:
            return "Inconsistent"

    combined_data["condition"] = combined_data.apply(get_condition, axis=1)
    result_df = combined_data.groupby(["participant_id", "session", "condition"]).agg(
        {
            "correct": "mean",
            "rt": "mean",
        }
    )
    result_df["ies"] = result_df["rt"] / result_df["correct"]
    pivot_ies = result_df.pivot_table(
        index=["participant_id", "session"],
        columns="condition",
        values="ies",
        aggfunc="mean",
    ).reset_index()

    interference_ies = pd.DataFrame(
        {
            "participant_id": pivot_ies["participant_id"],
            "session": pivot_ies["session"],
            "condition": "Inconsistent-Consistent",
            "ies": pivot_ies["Inconsistent"] - pivot_ies["Consistent"],
        }
    )

    incongruent_neutral_ies = pd.DataFrame(
        {
            "participant_id": pivot_ies["participant_id"],
            "session": pivot_ies["session"],
            "condition": "Inconsistent-Neutral",
            "ies": pivot_ies["Inconsistent"] - pivot_ies["Neutral"],
        }
    )

    neutral_congruent_ies = pd.DataFrame(
        {
            "participant_id": pivot_ies["participant_id"],
            "session": pivot_ies["session"],
            "condition": "Neutral-Consistent",
            "ies": pivot_ies["Neutral"] - pivot_ies["Consistent"],
        }
    )
    # Calculate interference effects for each participant and session
    # First, pivot the data to have conditions as columns for easier calculation
    pivot_acc = combined_data.pivot_table(
        index=["participant_id", "session"],
        columns="condition",
        values="correct",
        aggfunc="mean",
    ).reset_index()

    pivot_rt = combined_data.pivot_table(
        index=["participant_id", "session"],
        columns="condition",
        values="rt",
        aggfunc="mean",
    ).reset_index()

    # Calculate interference effects for accuracy
    interference_acc = pd.DataFrame(
        {
            "participant_id": pivot_acc["participant_id"],
            "session": pivot_acc["session"],
            "condition": "Inconsistent-Consistent",
            "correct": pivot_acc["Inconsistent"] - pivot_acc["Consistent"],
        }
    )

    incongruent_neutral_acc = pd.DataFrame(
        {
            "participant_id": pivot_acc["participant_id"],
            "session": pivot_acc["session"],
            "condition": "Inconsistent-Neutral",
            "correct": pivot_acc["Inconsistent"] - pivot_acc["Neutral"],
        }
    )

    neutral_congruent_acc = pd.DataFrame(
        {
            "participant_id": pivot_acc["participant_id"],
            "session": pivot_acc["session"],
            "condition": "Neutral-Consistent",
            "correct": pivot_acc["Neutral"] - pivot_acc["Consistent"],
        }
    )

    # Calculate interference effects for RT
    interference_rt = pd.DataFrame(
        {
            "participant_id": pivot_rt["participant_id"],
            "session": pivot_rt["session"],
            "condition": "Inconsistent-Consistent",
            "rt": pivot_rt["Inconsistent"] - pivot_rt["Consistent"],
        }
    )

    incongruent_neutral_rt = pd.DataFrame(
        {
            "participant_id": pivot_rt["participant_id"],
            "session": pivot_rt["session"],
            "condition": "Inconsistent-Neutral",
            "rt": pivot_rt["Inconsistent"] - pivot_rt["Neutral"],
        }
    )

    neutral_congruent_rt = pd.DataFrame(
        {
            "participant_id": pivot_rt["participant_id"],
            "session": pivot_rt["session"],
            "condition": "Neutral-Consistent",
            "rt": pivot_rt["Neutral"] - pivot_rt["Consistent"],
        }
    )

    # Combine all interference effects
    interference_effects = pd.merge(
        interference_acc, interference_rt, on=["participant_id", "session", "condition"]
    )
    interference_effects = pd.merge(
        interference_effects,
        interference_ies,
        on=["participant_id", "session", "condition"],
    )

    incongruent_neutral_effects = pd.merge(
        incongruent_neutral_acc,
        incongruent_neutral_rt,
        on=["participant_id", "session", "condition"],
    )
    incongruent_neutral_effects = pd.merge(
        incongruent_neutral_effects,
        incongruent_neutral_ies,
        on=["participant_id", "session", "condition"],
    )

    neutral_congruent_effects = pd.merge(
        neutral_congruent_acc,
        neutral_congruent_rt,
        on=["participant_id", "session", "condition"],
    )
    neutral_congruent_effects = pd.merge(
        neutral_congruent_effects,
        neutral_congruent_ies,
        on=["participant_id", "session", "condition"],
    )

    # Combine all effects with the original data
    interference_df = pd.concat(
        [
            interference_effects,
            incongruent_neutral_effects,
            neutral_congruent_effects,
        ]
    )

    # Create result dataframe
    result_df = result_df.reset_index()
    stim_duration = 500
    result_df["rt"] = result_df["rt"] * 1000 + stim_duration
    result_df.rename(columns={"correct": "accuracy"}, inplace=True)
    result_df["task"] = "stroop"
    interference_df["rt"] = interference_df["rt"] * 1000
    interference_df.rename(columns={"correct": "accuracy"}, inplace=True)
    interference_df["task"] = "stroop"
    result_df = split_emo_normal(result_df)
    interference_df = split_emo_normal(interference_df)
    return interference_df, result_df


# 提取BART任务的准确率和rt
def extract_bart(experiment_config):
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
    session_files = load_data(
        experiment_config["experiment_type"],
        experiment_config.get("data_path", "data"),
    )
    all_files_data = []

    for session, files in session_files.items():
        for file in files:
            df = pd.read_csv(file)
            # remove rows with any NA values
            df = df[df["exploded"].notna()]
            df = df[df["rt"].notna()]

            # Extract participant ID from filename
            file_name = os.path.basename(file)
            participant_id = int(file_name.split("_")[1])

            # Add session and participant information to the dataframe
            df["session"] = session
            df["participant_id"] = participant_id
            df["exploded"] = df["exploded"].astype(int)  # Ensure exploded is numeric

            all_files_data.append(df)

    # Combine all data
    if all_files_data:
        combined_data = pd.concat(all_files_data, ignore_index=True)
        # Calculate explosion rate and earnings metrics by participant and session
        result_df = (
            combined_data.groupby(["participant_id", "session"])
            .agg(
                explosion_rate=("exploded", lambda x: (sum(x) / len(x)) * 100),
                rt=("rt", "mean"),
                pumps=("pumps", "mean"),
                total_earned=("earned", "sum"),
                n_unexploded=("exploded", lambda x: sum(1 - x)),
                variance_first_half=("pumps", lambda x: np.var(x[: len(x) // 2])),
                variance_second_half=("pumps", lambda x: np.var(x[len(x) // 2 :])),
            )
            .reset_index()
        )
        result_df["mean_earned_per_unexploded"] = (
            result_df["total_earned"] / result_df["n_unexploded"]
        )
        result_df["rt"] = result_df["rt"] * 1000
        # Add task and condition columns
        result_df["task"] = "BART"
        # 将数据分为emo和normal
        result_df = split_emo_normal(result_df)
        print(
            f"Processed BART explosion rates for {len(result_df)} participant-session combinations"
        )

        return result_df
    else:
        # Return empty DataFrame with correct structure if no data
        return pd.DataFrame(
            {
                "task": [],
                "participant_id": [],
                "session": [],
                "condition": [],
                "explosion_rate": [],
            }
        )


def extract_emotion(experiment_config):
    """
    Extract valence and arousal ratings from emotion experiment data

    Parameters:
    -----------
    session_files : dict
        Dictionary containing session data files

    Returns:
    --------
    DataFrame : DataFrame containing participant ID, session, emotion type, and ratings
    """
    session_files = load_data(
        experiment_config["experiment_type"],
        experiment_config.get("data_path", "data"),
    )

    all_data = []
    for session, files in session_files.items():
        for file in files:
            df = pd.read_csv(file)

            # Extract participant ID from filename
            file_name = os.path.basename(file)
            participant_id = int(file_name.split("_")[1])

            # Create emotion type from category and phase
            df["emotion_type"] = df["category"]
            # For negative stimuli in regulation phase, mark as "regulation"
            df.loc[
                (df["category"] == "negative") & (df["phase"] == "regulation"),
                "emotion_type",
            ] = "regulation"
            df["session"] = session
            df["participant_id"] = participant_id
            all_data.append(df)

    combined_data = pd.concat(all_data, ignore_index=True)
    result_df = combined_data.groupby(
        ["participant_id", "session", "emotion_type"]
    ).agg(
        {
            "valence": "mean",
            "arousal": "mean",
        }
    )
    result_df.reset_index(inplace=True)
    result_df["task"] = "emotion"
    result_df = split_emo_normal(result_df)
    return result_df
