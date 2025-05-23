import matplotlib.pyplot as plt
import os
from utils.tasks import extract_emotion
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statannotations.Annotator import Annotator
from utils.plot import plot_bar_comparison


def plot_bar_significant(
    df,
    output_dir,
    x_var="emotion_type",
    y_var="valence",
    hue_var="session",
    row_var="group",
    title="",
    ylabel="",
    output_file=None,
):
    # 计算参与者人数
    participant_count = df["participant_id"].nunique()
    print(f"This group contains {participant_count} participants")
    participant_list = df["participant_id"].astype(str).unique().tolist()
    if row_var is not None:
        row_values = df[row_var].unique()
        fig, axes = plt.subplots(
            figsize=(8, 8),
            nrows=len(row_values),
            ncols=1,
        )
    else:
        row_values = [None]
        fig, ax = plt.subplots(figsize=(8, 8))
        axes = [ax]
    fig.suptitle(title)
    for row_value, ax in zip(row_values, axes):
        if row_value is not None:
            df_row = df[df[row_var] == row_value]
        else:
            df_row = df
        ax = plot_bar_comparison(
            df_row,
            x_var=x_var,
            y_var=y_var,
            title=title,
            ylabel=ylabel,
            ax=ax,
        )
        ax.set_title(row_value)
        # 添加显著性标记
        x_values = df[x_var].unique()
        hue_values = df[hue_var].unique()
        box_pairs = []
        for x in x_values:
            for i in range(len(hue_values)):
                for j in range(i + 1, len(hue_values)):
                    box_pairs.append(((x, hue_values[i]), (x, hue_values[j])))
        annotator = Annotator(
            ax,
            box_pairs,
            data=df,
            x=x_var,
            y=y_var,
            order=x_values,
            hue=hue_var,
        )
        annotator.configure(
            test="t-test_ind",
            text_format="star",
            loc="inside",
            hide_non_significant=True,
        )
        annotator.apply_and_annotate()
    if output_file is not None:
        fig.savefig(
            os.path.join(output_dir, "-".join(participant_list) + "_" + output_file)
        )
    plt.close()


def analyze_emotion():
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    emotion_config = {
        "experiment_type": "emotion",
    }

    # Extract earnings data
    exp_df = extract_emotion(emotion_config)
    exp_df = exp_df[
        # ~exp_df["participant_id"].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 19, 16, 25])
        ~exp_df["participant_id"].isin(
            [
                0,
                1,
                2,
                4,
                8,
            ]
        )
    ]

    # control_df = earnings_df[earnings_df["participant_id"] == 8]

    # 2 x 2 anova 检验condition和session对total_earned的影响
    # 使用 statsmodels 进行anova
    print("\n==== ANOVA valence ====")
    model = ols("valence ~ emotion_type * session * group", data=exp_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    print("\n==== ANOVA arousal ====")
    model = ols("arousal ~ emotion_type * session * group", data=exp_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # Perform  2x2x2 ANOVAs for negative-regulation
    print("\n==== 2x2x2 ANOVA for Negative-Regulation Emotion Type ====")
    df_negative_regulation = exp_df[
        exp_df["emotion_type"].isin(["negative", "regulation"])
    ].copy()

    try:
        # 2 (normal, emo) x 2 (pre-coffee, post-coffee) ANOVA for valence
        print("\n-- Valence for Negative-Regulation Emotion --")
        model_neg_val = ols(
            "valence ~ group * session * emotion_type", data=df_negative_regulation
        ).fit()
        anova_table_neg_val = sm.stats.anova_lm(model_neg_val, typ=2)
        print(anova_table_neg_val)

        # 2 (normal, emo) x 2 (pre-coffee, post-coffee) ANOVA for arousal
        print("\n-- Arousal for Negative-Regulation Emotion --")
        model_neg_aro = ols(
            "arousal ~ group * session * emotion_type", data=df_negative_regulation
        ).fit()
        anova_table_neg_aro = sm.stats.anova_lm(model_neg_aro, typ=2)
        print(anova_table_neg_aro)
    except Exception as e:
        print(f"Could not perform ANOVA on Negative-Regulation emotion type: {e}")

    # Perform  3x2x2 ANOVAs for negative-neutral-positive
    print("\n==== 3x2x2 ANOVA for Negative-Neutral-Positive Emotion Type ====")
    df_negative_neutral_positive = exp_df[
        exp_df["emotion_type"].isin(["negative", "neutral", "positive"])
    ].copy()

    try:
        # 3 (negative, neutral, positive) x 2 (pre-coffee, post-coffee) ANOVA for valence
        print("\n-- Valence for Negative-Neutral-Positive Emotion --")
        model_neg_neutral_pos_val = ols(
            "valence ~ group * session * emotion_type",
            data=df_negative_neutral_positive,
        ).fit()
        anova_table_neg_neutral_pos_val = sm.stats.anova_lm(
            model_neg_neutral_pos_val, typ=2
        )
        print(anova_table_neg_neutral_pos_val)

        # 3 (negative, neutral, positive) x 2 (pre-coffee, post-coffee) ANOVA for arousal
        print("\n-- Arousal for Negative-Neutral-Positive Emotion --")
        model_neg_neutral_pos_aro = ols(
            "arousal ~ group * session * emotion_type",
            data=df_negative_neutral_positive,
        ).fit()
        anova_table_neg_neutral_pos_aro = sm.stats.anova_lm(
            model_neg_neutral_pos_aro, typ=2
        )
        print(anova_table_neg_neutral_pos_aro)
    except Exception as e:
        print(f"Could not perform ANOVA on Negative-Neutral-Positive emotion type: {e}")

    # Create visualizations
    plot_bar_significant(
        exp_df,
        output_dir,
        row_var="group",
        x_var="emotion_type",
        y_var="valence",
        hue_var="session",
        title="Emotion Valence Comparison",
        ylabel="Valence",
        output_file="emotion_valence_comparison.png",
    )
    plot_bar_significant(
        exp_df,
        output_dir,
        row_var="group",
        x_var="emotion_type",
        y_var="arousal",
        hue_var="session",
        title="Emotion Arousal Comparison",
        ylabel="Arousal",
        output_file="emotion_arousal_comparison.png",
    )


if __name__ == "__main__":
    analyze_emotion()
