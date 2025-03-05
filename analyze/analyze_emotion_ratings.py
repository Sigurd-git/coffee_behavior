import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
from scipy import stats
from utils.plot import add_significance_markers

# Set font for displaying text
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False


def load_emotion_data(data_path="data"):
    """
    Load emotion experiment data files

    Parameters:
    -----------
    data_path : str
        Path to data directory

    Returns:
    --------
    dict : Dictionary containing session data files
    """
    session_files = {
        1: glob(os.path.join(data_path, "emotion_[0-9]*_session1.csv")),
        2: glob(os.path.join(data_path, "emotion_[0-9]*_session2.csv")),
    }

    print(f"Found {len(session_files[1])} files for session 1")
    print(f"Found {len(session_files[2])} files for session 2")

    return session_files


def extract_emotion_ratings(session_files):
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
    all_data = {
        "participant_id": [],
        "session": [],
        "emotion_type": [],
        "mean_valence": [],
        "mean_arousal": [],
    }

    for session, files in session_files.items():
        for file in files:
            try:
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

                # Process each emotion type separately
                for emotion_type in df["emotion_type"].unique():
                    emotion_data = df[df["emotion_type"] == emotion_type]

                    # Calculate mean valence and arousal for this emotion type
                    mean_valence = emotion_data["valence"].mean()
                    mean_arousal = emotion_data["arousal"].mean()

                    # Add data to results
                    all_data["participant_id"].append(participant_id)
                    all_data["session"].append(session)
                    all_data["emotion_type"].append(emotion_type)
                    all_data["mean_valence"].append(mean_valence)
                    all_data["mean_arousal"].append(mean_arousal)

                print(
                    f"Processed emotion ratings for participant {participant_id}, session {session}"
                )

            except Exception as e:
                print(f"Error processing file {file}: {e}")

    return pd.DataFrame(all_data)


def analyze_emotion_ratings_by_condition(df, measure):
    """
    Analyze emotion ratings by session and emotion type

    Parameters:
    -----------
    df : DataFrame
        DataFrame containing ratings data
    measure : str
        Column name for the measure to analyze (mean_valence or mean_arousal)

    Returns:
    --------
    DataFrame : Descriptive statistics for the measure
    """
    # Calculate descriptive statistics grouped by session and emotion type
    mean_stats = (
        df.groupby(["session", "emotion_type"])[measure]
        .agg(["mean"])
        .round(2)
        .reset_index()
    )
    std_stats = (
        df.groupby(["session", "emotion_type"])[measure]
        .agg(["std"])
        .round(2)
        .reset_index()
    )
    count_stats = (
        df.groupby(["session", "emotion_type"])[measure]
        .agg(["count"])
        .round(2)
        .reset_index()
    )

    desc_stats = pd.merge(mean_stats, std_stats, on=["session", "emotion_type"])
    desc_stats = pd.merge(desc_stats, count_stats, on=["session", "emotion_type"])
    return desc_stats


def run_emotion_analysis(df, measure):
    """
    Run a simplified analysis for session effect on each emotion type

    Parameters:
    -----------
    df : DataFrame
        DataFrame containing ratings data
    measure : str
        Column name for the measure to analyze (mean_valence or mean_arousal)

    Returns:
    --------
    dict : Results from t-tests for each emotion type
    """
    results = {}

    for emotion_type in df["emotion_type"].unique():
        # Filter data for this emotion type
        emotion_df = df[df["emotion_type"] == emotion_type]

        # Find participants who completed both sessions for this emotion type
        participants = set(
            emotion_df[emotion_df["session"] == 1]["participant_id"]
        ) & set(emotion_df[emotion_df["session"] == 2]["participant_id"])

        if not participants:
            print(f"No participants completed both sessions for {emotion_type}")
            continue

        # Filter data to include only these participants
        complete_data = emotion_df[emotion_df["participant_id"].isin(participants)]

        # Prepare paired data
        session1_data = []
        session2_data = []

        for pid in participants:
            s1_data = complete_data[
                (complete_data["session"] == 1)
                & (complete_data["participant_id"] == pid)
            ]
            s2_data = complete_data[
                (complete_data["session"] == 2)
                & (complete_data["participant_id"] == pid)
            ]

            if not s1_data.empty and not s2_data.empty:
                session1_data.append(s1_data[measure].iloc[0])
                session2_data.append(s2_data[measure].iloc[0])

        # Perform statistical test
        if len(session1_data) > 1:
            t_stat, p_val = stats.ttest_rel(session1_data, session2_data)
            results[emotion_type] = {
                "n_subjects": len(session1_data),
                "t_stat": t_stat,
                "p_val": p_val,
                "session1_mean": np.mean(session1_data),
                "session2_mean": np.mean(session2_data),
            }

    return results


def plot_emotion_ratings(df, measure, title, ylabel, output_file=None):
    """
    Plot emotion ratings comparison by session and emotion type

    Parameters:
    -----------
    df : DataFrame
        DataFrame containing ratings data
    measure : str
        Column name for the measure to plot
    title : str
        Plot title
    ylabel : str
        Y-axis label
    output_file : str
        Path to save the output figure
    """
    plt.figure(figsize=(12, 7))
    df.sort_values(by="emotion_type", inplace=True)
    # Create bar plot with session as x and emotion_type as hue
    ax = sns.barplot(
        x="emotion_type",
        y=measure,
        hue="session",
        data=df,
        estimator=np.mean,
        ci=68,
        palette=["lightgreen", "lightcoral", "lightblue", "pink"],
    )

    # Set labels and title
    plt.title(title)
    plt.xlabel("Experiment Phase")
    plt.ylabel(ylabel)

    # Add significance markers
    stats_results = run_emotion_analysis(df, measure)
    for i, emotion_type in enumerate(df["emotion_type"].unique()):
        if emotion_type in stats_results:
            result = stats_results[emotion_type]
            if result["p_val"] is not None and result["n_subjects"] > 1:
                # Calculate height for significance marker
                emotion_data = df[df["emotion_type"] == emotion_type]
                max_height = emotion_data.groupby("session")[measure].mean().max() * 1.1

                # Add markers for each emotion type
                # Adjust x positions based on bar positions
                x_positions = [i - 0.2, i + 0.2]
                add_significance_markers(ax, x_positions, result["p_val"], max_height)

    plt.tight_layout()

    # Save figure if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()


def analyze_emotion_ratings():
    """
    Analyze emotion ratings and compare between sessions and emotion types
    """
    print("Starting analysis of emotion experiment ratings data...")

    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    session_files = load_emotion_data()

    # Extract ratings data
    ratings_df = extract_emotion_ratings(session_files)

    if ratings_df.empty:
        print("No valid emotion experiment data found")
        return

    # 将数据分为实验组（除8号外）和对照组（8号）
    exp_df = ratings_df[ratings_df["participant_id"] != 8]
    control_df = ratings_df[ratings_df["participant_id"] == 8]

    print("\n==== Analyzing experimental group data (excluding subject 8) ====")
    analyze_group_data(exp_df, output_dir, "exp")

    print("\n==== Analyzing control group data (only subject 8) ====")
    analyze_group_data(control_df, output_dir, "control")


def analyze_group_data(df, output_dir, group_name):
    """
    Analyze emotion ratings for a specific group

    Parameters:
    -----------
    df : DataFrame
        DataFrame containing ratings data for the group
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

    # Analyze valence and arousal
    measures = {"mean_valence": "Valence Rating", "mean_arousal": "Arousal Rating"}

    for measure, measure_name in measures.items():
        print(f"\nAnalyzing {measure_name}...")

        # Calculate descriptive statistics
        desc_stats = analyze_emotion_ratings_by_condition(df, measure)

        print(f"\n{measure_name} descriptive statistics:")
        print(desc_stats)

        # Run statistical tests
        stats_results = run_emotion_analysis(df, measure)

        if stats_results:
            print(f"\n{measure_name} statistical analysis results:")

            emotion_type_labels = {
                "positive": "Positive",
                "negative": "Negative",
                "neutral": "Neutral",
                "regulation": "Negative Regulation",
            }

            for emotion_type, results in stats_results.items():
                emotion_name = emotion_type_labels.get(emotion_type, emotion_type)
                print(f"\n{emotion_name} emotion type:")
                print(
                    f"Number of subjects completing both sessions: {results['n_subjects']}"
                )
                print(f"t = {results['t_stat']:.3f}, p = {results['p_val']:.3f}")
                print(f"Pre-coffee mean: {results['session1_mean']:.2f}")
                print(f"Post-coffee mean: {results['session2_mean']:.2f}")

        # Create visualization
        plot_title = (
            f"{group_name.upper()} Group Emotion Experiment {measure_name} Comparison"
        )
        output_file = os.path.join(output_dir, f"emotion_{group_name}_{measure}.png")

        plot_emotion_ratings(df, measure, plot_title, measure_name, output_file)


if __name__ == "__main__":
    analyze_emotion_ratings()
