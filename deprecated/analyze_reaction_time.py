import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.plot import plot_bar_comparison
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
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial"]
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
            try:
                df = pd.read_csv(file)
            except FileNotFoundError:
                print(f"Warning: File not found {file}, skipping.")
                continue
            except pd.errors.EmptyDataError:
                print(f"Warning: File is empty {file}, skipping.")
                continue
            except Exception as e:
                print(f"Error reading file {file}: {e}, skipping.")
                continue

            # Check for required columns
            required_cols = ["phase", "rating_rt", "participant_id"]
            if not all(col in df.columns for col in required_cols):
                print(
                    f"Warning: File {file} missing required columns ({required_cols}), skipping."
                )
                continue

            # Process each phase separately (baseline and regulation)
            for phase in df["phase"].unique():
                phase_data = df[df["phase"] == phase]

                # Calculate mean reaction time using rating_rt (as specified)
                # Ensure rt is numeric and handle potential errors
                try:
                    valid_rts = phase_data[
                        phase_data["rating_rt"].notna()
                        & pd.to_numeric(
                            phase_data["rating_rt"], errors="coerce"
                        ).notna()
                        & (phase_data["rating_rt"] > 0)
                    ]["rating_rt"]
                except Exception as e:
                    print(
                        f"Error processing rating_rt in {file} for phase {phase}: {e}"
                    )
                    continue  # Skip this phase if rt processing fails

                if not valid_rts.empty:
                    # Ensure participant_id is available
                    if df["participant_id"].empty:
                        print(
                            f"Warning: No participant ID found in {file}, skipping phase {phase}."
                        )
                        continue

                    rt = (
                        valid_rts.astype(float).mean() * 1000
                    )  # Convert to milliseconds

                    all_data["task"].append("emotion")
                    all_data["participant_id"].append(df["participant_id"].iloc[0])
                    all_data["session"].append(session)
                    all_data["condition"].append(phase)
                    all_data["rt"].append(rt)

    if not all_data["participant_id"]:
        print("Warning: No valid emotion regulation data found across all files.")
        return pd.DataFrame()  # Return empty DataFrame if no data

    return pd.DataFrame(all_data)


def extract_stroop_detailed_metrics(data_path="data", exclude_participant=None):
    """
    从Stroop实验数据中提取详细的反应时和正确率指标
    """
    # Use glob to find files, ensure path is correct
    pattern = os.path.join(data_path, "stroop_*_session*_stats_new.csv")
    all_files = sorted(glob(pattern))

    # Filter out excluded participants if specified
    if exclude_participant:
        exclude_participant_str = [
            str(pid) for pid in np.atleast_1d(exclude_participant)
        ]
        files_to_process = [
            f
            for f in all_files
            if not any(
                f"_{pid}_" in os.path.basename(f) for pid in exclude_participant_str
            )
        ]
        print(f"Excluding participants: {exclude_participant}")
    else:
        files_to_process = all_files

    # Separate files by session based on filename
    session1_files = sorted([f for f in files_to_process if "_session1_" in f])
    session2_files = sorted([f for f in files_to_process if "_session2_" in f])

    print(f"Found {len(session1_files)} files for Session 1.")
    # print("Session 1 files:", session1_files) # Optionally print file list
    print(f"Found {len(session2_files)} files for Session 2.")
    # print("Session 2 files:", session2_files) # Optionally print file list

    all_data_list = []  # Use a list of dicts for efficiency

    # Define expected columns and their corresponding keys in the output dict
    metric_mapping = {
        "总体": {"accuracy": "正确率", "rt": "平均反应时"},
        "一致": {"accuracy": "一致条件正确率", "rt": "一致条件反应时"},
        "不一致": {"accuracy": "不一致条件正确率", "rt": "不一致条件反应时"},
        "中性": {"accuracy": "中性词正确率", "rt": "中性词反应时"},
    }

    # Processing function for a single file to reduce redundancy
    def process_file(file, session):
        try:
            df = pd.read_csv(file)
            if df.empty:
                print(f"Warning: File is empty {file}, skipping.")
                return None
            if "participant_id" not in df.columns or df["participant_id"].isna().all():
                print(
                    f"Warning: Missing or invalid participant_id in {file}, skipping."
                )
                return None

            participant_id = df["participant_id"].iloc[0]
            # print(f"\nProcessing file: {os.path.basename(file)} - Participant ID: {participant_id}")

            # Only process the 'total' level row
            total_data_row = df[df["level"] == "total"]
            if total_data_row.empty:
                print(
                    f"Warning: No 'total' level data found in {file} for participant {participant_id}, skipping."
                )
                return None
            total_data = total_data_row.iloc[0]

            file_results = []
            # Extract metrics for defined conditions
            for condition, keys in metric_mapping.items():
                acc_key = keys["accuracy"]
                rt_key = keys["rt"]
                if acc_key not in total_data or rt_key not in total_data:
                    print(
                        f"Warning: Missing keys '{acc_key}' or '{rt_key}' for condition '{condition}' in {file}, skipping condition."
                    )
                    continue

                accuracy = pd.to_numeric(total_data[acc_key], errors="coerce")
                rt = pd.to_numeric(total_data[rt_key], errors="coerce")

                if pd.isna(accuracy) or pd.isna(rt):
                    print(
                        f"Warning: Non-numeric data for accuracy or rt for condition '{condition}' in {file}, skipping condition."
                    )
                    continue

                file_results.append(
                    {
                        "participant_id": participant_id,
                        "session": session,
                        "condition": condition,
                        "accuracy": accuracy,
                        "rt": rt,
                    }
                )

            # Calculate and add difference metrics if base conditions exist
            conditions_present = {res["condition"]: res for res in file_results}
            if "不一致" in conditions_present and "中性" in conditions_present:
                incongruent_acc = conditions_present["不一致"]["accuracy"]
                neutral_acc = conditions_present["中性"]["accuracy"]
                incongruent_rt = conditions_present["不一致"]["rt"]
                neutral_rt = conditions_present["中性"]["rt"]
                file_results.append(
                    {
                        "participant_id": participant_id,
                        "session": session,
                        "condition": "不一致-中性",
                        "accuracy": incongruent_acc - neutral_acc,
                        "rt": incongruent_rt - neutral_rt,
                    }
                )
            if "一致" in conditions_present and "中性" in conditions_present:
                congruent_acc = conditions_present["一致"]["accuracy"]
                neutral_acc = conditions_present["中性"]["accuracy"]
                congruent_rt = conditions_present["一致"]["rt"]
                neutral_rt = conditions_present["中性"]["rt"]
                file_results.append(
                    {
                        "participant_id": participant_id,
                        "session": session,
                        "condition": "一致-中性",
                        "accuracy": congruent_acc - neutral_acc,
                        "rt": congruent_rt - neutral_rt,
                    }
                )

            return file_results

        except FileNotFoundError:
            print(f"Warning: File not found {file}, skipping.")
            return None
        except pd.errors.EmptyDataError:
            print(f"Warning: File is empty {file}, skipping.")
            return None
        except Exception as e:
            print(f"Error processing file {file}: {e}, skipping.")
            return None

    # 处理所有文件
    for file in session1_files:
        result = process_file(file, 1)
        if result:
            all_data_list.extend(result)
    for file in session2_files:
        result = process_file(file, 2)
        if result:
            all_data_list.extend(result)

    if not all_data_list:
        print("\nWarning: No valid detailed Stroop data extracted.")
        return pd.DataFrame()  # Return empty DataFrame if no data

    all_df = pd.DataFrame(all_data_list)

    # 打印找到的所有被试ID
    unique_participants = sorted(all_df["participant_id"].unique())
    print(
        f"\nProcessed data for {len(unique_participants)} unique participants:",
        unique_participants,
    )

    return all_df


def analyze_stroop_detailed(exclude_participant=None):
    """分析Stroop任务在不同条件下的反应时、正确率和IES"""

    # 提取数据, pass the exclusion list directly
    df = extract_stroop_detailed_metrics(exclude_participant=exclude_participant)

    if df.empty:
        print("未找到有效的Stroop任务数据进行详细分析")
        return

    # 创建输出目录
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Filter for primary conditions needed for IES/Effect calculation
    primary_conditions = ["一致", "不一致", "中性"]
    analysis_df = df[df["condition"].isin(primary_conditions)].copy()

    # Ensure accuracy is not zero and rt is positive before calculating IES
    analysis_df = analysis_df[(analysis_df["accuracy"] > 0) & (analysis_df["rt"] > 0)]

    # Calculate IES (RT / (Accuracy / 100)) = RT * 100 / Accuracy
    # Handle potential division by zero or invalid values gracefully
    analysis_df["ies"] = np.where(
        analysis_df["accuracy"] > 0,
        (analysis_df["rt"] * 100) / analysis_df["accuracy"],
        np.nan,
    )  # Assign NaN if accuracy is 0

    # Remove rows where IES could not be calculated
    analysis_df.dropna(subset=["ies"], inplace=True)

    # Calculate descriptive statistics for RT, Accuracy, and IES
    desc_stats = (
        analysis_df.groupby(["session", "condition"])
        .agg(
            rt_mean=("rt", "mean"),
            rt_std=("rt", "std"),
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            ies_mean=("ies", "mean"),
            ies_std=("ies", "std"),
            count=("participant_id", "count"),  # Count valid entries per group
        )
        .round(2)
    )

    print("描述性统计 (RT, Accuracy, IES):")
    print(desc_stats)

    # Find participants present in both sessions AFTER filtering and IES calculation
    pivoted_check = analysis_df.pivot_table(
        index="participant_id", columns="session", values="ies", aggfunc="size"
    )
    common_participants = pivoted_check.dropna().index.tolist()

    print(
        f"\nNumber of participants with valid data in both sessions: {len(common_participants)}"
    )
    if not common_participants:
        print(
            "No participants completed both sessions with valid data for paired analysis."
        )
        return

    # Filter the dataframe to include only common participants
    paired_df = analysis_df[
        analysis_df["participant_id"].isin(common_participants)
    ].copy()

    # Pivot table to get measures per participant per session
    pivoted_metrics = paired_df.pivot_table(
        index="participant_id",
        columns=["session", "condition"],
        values=["rt", "ies"],  # Pivot RT and IES
    )

    # Calculate interference effects (RT and IES)
    results_data = []
    for pid in common_participants:
        for session in [1, 2]:
            subj_session_data = paired_df[
                (paired_df["participant_id"] == pid) & (paired_df["session"] == session)
            ]
            # Create a dictionary for quick lookup of metrics for this participant/session
            metrics = subj_session_data.set_index("condition").to_dict()

            # Check if all necessary conditions are present
            required = ["不一致", "一致", "中性"]
            if not all(c in metrics["rt"] for c in required):
                print(
                    f"Warning: Participant {pid} Session {session} missing conditions for effect calculation. Skipping."
                )
                continue

            # RT Effects
            incongruent_congruent_rt = metrics["rt"]["不一致"] - metrics["rt"]["一致"]
            incongruent_neutral_rt = metrics["rt"]["不一致"] - metrics["rt"]["中性"]
            neutral_congruent_rt = metrics["rt"]["中性"] - metrics["rt"]["一致"]

            # IES Effects
            incongruent_congruent_ies = (
                metrics["ies"]["不一致"] - metrics["ies"]["一致"]
            )
            incongruent_neutral_ies = metrics["ies"]["不一致"] - metrics["ies"]["中性"]
            neutral_congruent_ies = metrics["ies"]["中性"] - metrics["ies"]["一致"]

            results_data.append(
                {
                    "Subject_ID": pid,
                    "Session": session,  # Keep session as number for easier pivoting
                    "Incongruent-Congruent_RT": round(incongruent_congruent_rt, 2),
                    "Incongruent-Neutral_RT": round(incongruent_neutral_rt, 2),
                    "Neutral-Congruent_RT": round(neutral_congruent_rt, 2),
                    "Incongruent-Congruent_IES": round(incongruent_congruent_ies, 2),
                    "Incongruent-Neutral_IES": round(incongruent_neutral_ies, 2),
                    "Neutral-Congruent_IES": round(neutral_congruent_ies, 2),
                }
            )

    if not results_data:
        print("Could not calculate interference effects for any participant.")
        return

    # Create DataFrame of interference effects
    results_df = pd.DataFrame(results_data)

    # Pivot the results to have sessions side-by-side
    pivot_interference_df = results_df.pivot(
        index="Subject_ID",
        columns="Session",
        values=[
            "Incongruent-Congruent_RT",
            "Incongruent-Neutral_RT",
            "Neutral-Congruent_RT",
            "Incongruent-Congruent_IES",
            "Incongruent-Neutral_IES",
            "Neutral-Congruent_IES",
        ],
    )

    # Flatten MultiIndex columns for easier access and saving
    pivot_interference_df.columns = [
        f"{val}_Session{col}" for val, col in pivot_interference_df.columns
    ]

    # Calculate descriptive statistics for the interference effects
    interference_desc_stats = pivot_interference_df.agg(["mean", "std"]).round(2)

    # Add descriptive stats as rows (optional, can keep separate)
    final_interference_df = pd.concat([pivot_interference_df, interference_desc_stats])
    final_interference_df.index.name = "Subject_ID / Stat"  # Clarify index

    # Perform paired t-tests between sessions for each interference measure
    t_test_results = []
    measures = [
        "Incongruent-Congruent_RT",
        "Incongruent-Neutral_RT",
        "Neutral-Congruent_RT",
        "Incongruent-Congruent_IES",
        "Incongruent-Neutral_IES",
        "Neutral-Congruent_IES",
    ]

    for measure in measures:
        session1_col = f"{measure}_Session1"
        session2_col = f"{measure}_Session2"

        # Ensure columns exist before accessing
        if (
            session1_col not in pivot_interference_df.columns
            or session2_col not in pivot_interference_df.columns
        ):
            print(
                f"Warning: Columns missing for t-test on measure {measure}. Skipping t-test."
            )
            continue

        session1_data = pivot_interference_df[session1_col].dropna()
        session2_data = pivot_interference_df[session2_col].dropna()

        # Ensure data is aligned by participant for paired test (it should be by pivot construction)
        aligned_s1 = session1_data.reindex(common_participants)
        aligned_s2 = session2_data.reindex(common_participants)
        valid_indices = aligned_s1.notna() & aligned_s2.notna()
        aligned_s1 = aligned_s1[valid_indices]
        aligned_s2 = aligned_s2[valid_indices]

        if len(aligned_s1) < 2:
            print(
                f"Warning: Insufficient paired data (<2) for t-test on measure {measure}. Skipping t-test."
            )
            t_stat, p_val = np.nan, np.nan  # Assign NaN if test cannot be performed
        else:
            t_stat, p_val = stats.ttest_rel(aligned_s1, aligned_s2)

        t_test_results.append(
            {
                "Measure": measure,
                "t_statistic": round(t_stat, 3) if not np.isnan(t_stat) else "N/A",
                "p_value": round(p_val, 3) if not np.isnan(p_val) else "N/A",
                "Session1_Mean": round(aligned_s1.mean(), 2),
                "Session1_SD": round(aligned_s1.std(), 2),
                "Session2_Mean": round(aligned_s2.mean(), 2),
                "Session2_SD": round(aligned_s2.std(), 2),
                "N_pairs": len(aligned_s1),
            }
        )

    t_test_df = pd.DataFrame(t_test_results)

    # Save results to Excel
    excel_path = os.path.join(output_dir, "stroop_detailed_analysis.xlsx")
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            desc_stats.to_excel(writer, sheet_name="Descriptive Stats (Raw)")
            final_interference_df.to_excel(writer, sheet_name="Interference Effects")
            t_test_df.to_excel(
                writer, sheet_name="Paired T-tests (Interference)", index=False
            )
        print(f"\nDetailed Stroop analysis results saved to: {excel_path}")
    except Exception as e:
        print(f"\nError saving detailed Stroop analysis to Excel: {e}")

    # Print results preview
    print("Interference Effects Preview (Participants x Session):")
    print(pivot_interference_df.round(2).head())  # Show first few participants

    print("Descriptive Statistics (Interference Effects):")
    print(interference_desc_stats)

    print("Paired t-test Results (Session1 vs Session2 on Interference Effects):")
    print(t_test_df.to_string(index=False))


# 示例用法
if __name__ == "__main__":
    print("Starting analysis...")

    # 创建输出目录
    os.makedirs("output", exist_ok=True)

    # Define participants to exclude from all analyses run here
    master_exclude_ids = [0, 1, 2, 4, 8]  # Example exclusion list
    # master_exclude_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 19, 16, 25] # Alternative

    # === Run Detailed Stroop Analysis ===
    print("==== Analyzing Detailed Stroop Metrics ====")
    analyze_stroop_detailed(exclude_participant=master_exclude_ids)

    # === Example: Run Emotion RT Analysis ===
    # This requires file patterns to be defined correctly for your data structure
    # emotion_config = {
    #     "experiment_type": "emotion",
    #     "extract_function": extract_emotion_rt, # Assumes this function is defined above
    #     # Correctly define file patterns based on your data location and naming
    #     "session_files": {
    #          1: sorted(glob("data/emotion_*_session1.csv")), # Adjust glob pattern as needed
    #          2: sorted(glob("data/emotion_*_session2.csv"))  # Adjust glob pattern as needed
    #     },
    #     # Plotting config (optional, needs a dedicated analysis function now)
    #     # "plot_config": {
    #     #     "x_var": "condition",
    #     #     "title": "Emotion Task Rating RT Comparison",
    #     #     "xlabel": "Experiment Phase",
    #     #     "ylabel": "Rating Reaction Time (ms)",
    #     #     "output_file": "emotion_rt_comparison.png",
    #     # },
    #     "output_dir": "output",
    # }
    # print("\n==== Analyzing Emotion RT ====")
    # # Extract data first
    # emotion_df = extract_emotion_rt(emotion_config["session_files"])
    # if not emotion_df.empty:
    #      # Filter excluded participants
    #      emotion_df_filtered = emotion_df[~emotion_df["participant_id"].isin(master_exclude_ids)]
    #      print(f"Emotion RT data extracted for {len(emotion_df_filtered['participant_id'].unique())} participants after exclusion.")
    #      # Further analysis/plotting would go here, potentially in a new function
    #      # e.g., plot_emotion_rt(emotion_df_filtered, emotion_config)
    # else:
    #      print("No emotion RT data found or extracted.")

    print("Analysis script finished.")

    # --- Cleaned up section ---
    # The following configurations and calls are removed as the analysis
    # logic has been moved to analyze_stroop.py and analyze_nback.py

    # # N-back实验配置 (Removed)
    # nback_config = { ... }

    # # Stroop实验配置 (Removed - basic config, detailed is above)
    # stroop_config = { ... }

    # # 情绪实验配置 (Removed - basic config, example extraction is above)
    # emotion_config = { ... }

    # # 所有实验配置 (Removed)
    # config_list = [stroop_config] # Was only stroop previously

    # # 分析实验组数据 (Removed)
    # print("==== Analyzing experimental group data (excluding subject 8) ====")
    # ... (loop to modify config removed)
    # print("Starting analysis of individual experiments for experimental group...")
    # exp_results = analyze_combined_experiments(
    #     config_list,
    #     exclude_participant=master_exclude_ids,
    # )

    # # 分析对照组数据 (Removed)
    # print("\n==== Analyzing control group data (only subject 8) ====")
    # ... (control group logic removed)
