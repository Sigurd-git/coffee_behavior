import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.plot import (
    plot_bar_comparison,
    paired_t_test,
    add_significance_markers,
)
from utils.tasks import (
    extract_nback,
    extract_stroop,
    extract_bart,
)
from statannotations.Annotator import Annotator
from statsmodels.formula.api import ols
import statsmodels.api as sm


def analyze_experiment_accuracy(
    experiment_config, exclude_participant=None, only_participant=None
):
    """
    通用实验正确率分析主函数

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
    extract_func = experiment_config["extract_function"]
    df = extract_func(experiment_config)
    if isinstance(df, tuple):
        df, original_df = df
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
            f"Experiment {experiment_config['experiment_type']} Accuracy analysis has no valid data after filtering"
        )
        return results  # Return early if no data

    print("\n==== ANOVA accuracy ====")
    try:
        model = ols("accuracy ~ condition * session * group", data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)
    except Exception as e:
        print(f"Could not perform ANOVA on Accuracy: {e}")

    # 6. 可视化
    if experiment_config.get("plot", True):
        plot_config = experiment_config.get("plot_config", {})
        row_var = plot_config.get("row_var", None)

        # Determine number of rows for subplots
        if row_var and row_var in df.columns:
            # Sort row values if they are sortable (e.g., numeric or standard strings)
            try:
                row_values = sorted(df[row_var].unique())
            except TypeError:
                row_values = df[
                    row_var
                ].unique()  # Keep original order if sorting fails
            n_rows = len(row_values)
        else:
            row_values = [None]
            n_rows = 1

        fig, axes = plt.subplots(
            figsize=plot_config.get(
                "figsize", (8, 5 * n_rows)
            ),  # Adjust height based on rows
            nrows=n_rows,
            ncols=1,
            squeeze=False,  # Always return 2D array for axes
        )
        axes = axes.flatten()  # Flatten to 1D array

        title = plot_config.get(
            "title",
            f"{experiment_config['experiment_type']} Task Accuracy Comparison",
        )
        fig.suptitle(title, y=1.02)  # Adjust title position slightly

        for i, row_value in enumerate(row_values):
            ax = axes[i]
            if row_value is not None:
                df_row = df[
                    df[row_var] == row_value
                ].copy()  # Use copy to avoid SettingWithCopyWarning
            else:
                df_row = df.copy()

            if df_row.empty:
                print(f"No Accuracy data to plot for row: {row_value}")
                ax.set_title(f"{row_value} (No Data)")
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue

            plot_bar_comparison(
                df_row,
                x_var=plot_config.get("x_var", "condition"),
                y_var=plot_config.get(
                    "y_var", "accuracy"
                ),  # Use accuracy as y-axis variable
                hue_var=plot_config.get("hue_var", "session"),
                xlabel=plot_config.get("xlabel", "Condition"),
                ylabel=plot_config.get("ylabel", "Accuracy (%)"),
                palette=plot_config.get("palette", ["lightblue", "lightgreen"]),
                ax=ax,
                show_values=plot_config.get("show_values", True),
            )
            if row_value is not None:
                ax.set_title(row_value)  # Set title for subplot if using row_var

            # 添加显著性标记
            x_var = plot_config.get("x_var", "condition")
            y_var = plot_config.get("y_var", "accuracy")  # Use correct y variable name
            hue_var = plot_config.get("hue_var", "session")

            # Ensure columns exist before proceeding
            if not all(col in df_row.columns for col in [x_var, y_var, hue_var]):
                print(
                    f"Skipping Accuracy annotations for {row_value}: Missing required columns."
                )
                continue

            try:
                x_values = sorted(df_row[x_var].unique())
                hue_values = sorted(df_row[hue_var].unique())
            except TypeError:
                x_values = df_row[x_var].unique()
                hue_values = df_row[hue_var].unique()
            box_pairs = []

            # Check if there are enough data points for paired t-test
            valid_pairs = True
            if len(hue_values) > 1:  # Need at least two hue levels for comparison
                for x in x_values:
                    # Check data for the first pair of hue values
                    group1_data = df_row[
                        (df_row[x_var] == x) & (df_row[hue_var] == hue_values[0])
                    ][y_var]
                    group2_data = df_row[
                        (df_row[x_var] == x) & (df_row[hue_var] == hue_values[1])
                    ][y_var]

                    # Ensure paired data has the same length and more than 1 point
                    if len(group1_data) != len(group2_data) or len(group1_data) < 2:
                        print(
                            f"Skipping Accuracy annotation for '{x}' in '{row_value}': Insufficient or unequal data points ({len(group1_data)} vs {len(group2_data)}) for paired t-test."
                        )
                        valid_pairs = False
                    # No need to create pair for this x, but continue checking others
                    else:
                        # Only add pair if data is valid
                        box_pairs.append(((x, hue_values[0]), (x, hue_values[1])))
            else:
                valid_pairs = (
                    False  # Cannot perform paired tests with only one hue level
                )
                print(
                    f"Skipping Accuracy annotations for {row_value}: Only one hue level present."
                )

            if box_pairs and valid_pairs:  # Only annotate if valid pairs exist
                annotator = Annotator(
                    ax,
                    box_pairs,
                    data=df_row,
                    x=x_var,
                    y=y_var,
                    order=x_values,
                    hue=hue_var,
                    hue_order=hue_values,  # Specify hue order
                )
                annotator.configure(
                    test="t-test_paired",
                    text_format="star",
                    loc="inside",
                    comparisons_correction=None,  # No correction for multiple tests here, consider if needed
                    hide_non_significant=True,
                    verbose=0,  # Reduce verbosity
                )
                try:
                    annotator.apply_and_annotate()
                except Exception as e:
                    print(f"Error applying Accuracy annotations for {row_value}: {e}")

        plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to prevent title overlap

        # 保存图表
        output_dir = experiment_config.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            plot_config.get(
                "output_file",
                f"{experiment_config['experiment_type']}_accuracy_comparison.png",
            ),
        )
        try:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Saved Accuracy plot to: {output_file}")
        except Exception as e:
            print(f"Failed to save Accuracy plot {output_file}: {e}")
        plt.close(fig)  # Close the figure to free memory

    return results


def analyze_nback_rt(
    experiment_config, exclude_participant=None, only_participant=None
):
    """
    Analyze reaction time for the N-back experiment.

    Parameters:
    -----------
    experiment_config : dict
        Configuration dictionary for the experiment.
    exclude_participant : list, optional
        List of participant IDs to exclude.
    only_participant : list, optional
        List of participant IDs to include exclusively.

    Returns:
    --------
    dict : Analysis results dictionary (currently empty).
    """
    results = {}

    # 2. Extract data
    extract_func = experiment_config["extract_function"]
    df = extract_func(experiment_config)
    if isinstance(df, tuple):
        df, original_df = df  # Assuming extract_func might return original df too

    # 3. Filter data based on participant IDs
    if exclude_participant is not None:
        df = df[~df["participant_id"].isin(np.atleast_1d(exclude_participant))]

    if only_participant is not None:
        df = df[df["participant_id"].isin(np.atleast_1d(only_participant))]

    # If there's no data after filtering, skip this experiment
    if df.empty:
        print(
            f"Experiment {experiment_config['experiment_type']} RT analysis has no valid data after filtering"
        )
        return results  # Return early if no data

    print("\n==== ANOVA Reaction Time (RT) ====")
    try:
        model = ols("rt ~ condition * session * group", data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)
    except Exception as e:
        print(f"Could not perform ANOVA on RT: {e}")

    # 6. Visualization
    if experiment_config.get("plot", True):
        plot_config = experiment_config.get("plot_config", {})
        row_var = plot_config.get("row_var", None)

        # Determine number of rows for subplots
        if row_var and row_var in df.columns:
            # Sort row values if they are sortable (e.g., numeric or standard strings)
            try:
                row_values = sorted(df[row_var].unique())
            except TypeError:
                row_values = df[
                    row_var
                ].unique()  # Keep original order if sorting fails
            n_rows = len(row_values)
        else:
            row_values = [None]
            n_rows = 1

        fig, axes = plt.subplots(
            figsize=plot_config.get(
                "figsize", (8, 5 * n_rows)
            ),  # Adjust height based on rows
            nrows=n_rows,
            ncols=1,
            squeeze=False,  # Always return 2D array for axes
        )
        axes = axes.flatten()  # Flatten to 1D array

        title = plot_config.get(
            "title",
            f"{experiment_config['experiment_type']} Task Reaction Time Comparison",
        )
        fig.suptitle(title, y=1.02)  # Adjust title position slightly

        for i, row_value in enumerate(row_values):
            ax = axes[i]
            if row_value is not None:
                df_row = df[
                    df[row_var] == row_value
                ].copy()  # Use copy to avoid SettingWithCopyWarning
            else:
                df_row = df.copy()

            if df_row.empty:
                print(f"No RT data to plot for row: {row_value}")
                ax.set_title(f"{row_value} (No Data)")
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue

            plot_bar_comparison(
                df_row,
                x_var=plot_config.get("x_var", "condition"),
                y_var=plot_config.get("y_var", "rt"),  # Use RT as y-axis variable
                hue_var=plot_config.get("hue_var", "session"),
                xlabel=plot_config.get("xlabel", "Condition"),
                ylabel=plot_config.get(
                    "ylabel", "Reaction Time (ms)"
                ),  # Correct Y label
                palette=plot_config.get("palette", ["lightblue", "lightgreen"]),
                ax=ax,
                show_values=plot_config.get("show_values", True),
            )
            if row_value is not None:
                ax.set_title(row_value)  # Set title for subplot if using row_var

            # Add significance markers
            x_var = plot_config.get("x_var", "condition")
            y_var = plot_config.get("y_var", "rt")
            hue_var = plot_config.get("hue_var", "session")

            # Ensure columns exist before proceeding
            if not all(col in df_row.columns for col in [x_var, y_var, hue_var]):
                print(
                    f"Skipping RT annotations for {row_value}: Missing required columns."
                )
                continue

            try:
                x_values = sorted(df_row[x_var].unique())
                hue_values = sorted(df_row[hue_var].unique())
            except TypeError:
                x_values = df_row[x_var].unique()
                hue_values = df_row[hue_var].unique()

            box_pairs = []

            # Check if there are enough data points for paired t-test
            valid_pairs = True
            if len(hue_values) > 1:  # Need at least two hue levels for comparison
                for x in x_values:
                    # Check data for the first pair of hue values
                    group1_data = df_row[
                        (df_row[x_var] == x) & (df_row[hue_var] == hue_values[0])
                    ][y_var]
                    group2_data = df_row[
                        (df_row[x_var] == x) & (df_row[hue_var] == hue_values[1])
                    ][y_var]

                    # Ensure paired data has the same length and more than 1 point
                    if len(group1_data) != len(group2_data) or len(group1_data) < 2:
                        print(
                            f"Skipping RT annotation for '{x}' in '{row_value}': Insufficient or unequal data points ({len(group1_data)} vs {len(group2_data)}) for paired t-test."
                        )
                        valid_pairs = False
                    # No need to create pair for this x, but continue checking others
                    else:
                        # Only add pair if data is valid
                        box_pairs.append(((x, hue_values[0]), (x, hue_values[1])))
            else:
                valid_pairs = (
                    False  # Cannot perform paired tests with only one hue level
                )
                print(
                    f"Skipping RT annotations for {row_value}: Only one hue level present."
                )

            if box_pairs and valid_pairs:  # Only annotate if valid pairs exist
                annotator = Annotator(
                    ax,
                    box_pairs,
                    data=df_row,
                    x=x_var,
                    y=y_var,
                    order=x_values,
                    hue=hue_var,
                    hue_order=hue_values,  # Specify hue order
                )
                annotator.configure(
                    test="t-test_paired",
                    text_format="star",
                    loc="inside",
                    comparisons_correction=None,  # No correction for multiple tests here, consider if needed
                    hide_non_significant=True,
                    verbose=0,  # Reduce verbosity
                )
                try:
                    annotator.apply_and_annotate()
                except Exception as e:
                    print(f"Error applying RT annotations for {row_value}: {e}")

        plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to prevent title overlap

        # Save the plot
        output_dir = experiment_config.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            plot_config.get(
                "output_file",
                f"{experiment_config['experiment_type']}_rt_comparison.png",  # Default filename for RT
            ),
        )
        try:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Saved RT plot to: {output_file}")
        except Exception as e:
            print(f"Failed to save RT plot {output_file}: {e}")
        plt.close(fig)  # Close the figure to free memory

    return results


# 示例用法
if __name__ == "__main__":
    # N-back Accuracy analysis configuration
    nback_acc_config = {
        "experiment_type": "nback",
        "extract_function": extract_nback,
        "plot_config": {
            "x_var": "condition",
            "y_var": "accuracy",  # Specify accuracy as the y variable
            "row_var": "group",
            "title": "N-back Task Accuracy Comparison",
            "xlabel": "Condition",
            "ylabel": "Accuracy (%)",
            "output_file": "nback_accuracy_comparison.png",
        },
        "output_dir": "output",
    }

    # N-back Reaction Time (RT) analysis configuration
    nback_rt_config = {
        "experiment_type": "nback",
        "extract_function": extract_nback,
        "plot_config": {
            "x_var": "condition",
            "y_var": "rt",  # Specify rt as the y variable
            "row_var": "group",
            "title": "N-back Task Reaction Time Comparison",
            "xlabel": "Condition",
            "ylabel": "Reaction Time (ms)",  # Correct y label for RT
            "output_file": "nback_rt_comparison.png",  # Specific output file for RT
        },
        "output_dir": "output",
    }

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Define participants to exclude
    exclude_ids = [0, 1, 2, 4, 8]
    # exclude_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 19, 25] # Alternative exclusion list

    # Run Accuracy analysis
    print("\n==== Starting N-back Accuracy Analysis ====")
    acc_results = analyze_experiment_accuracy(
        nback_acc_config,
        exclude_participant=exclude_ids,
    )
    print("N-back Accuracy analysis completed!")

    # # 分析对照组数据（只有8号被试） - 这部分逻辑需要更新，如果需要单独分析对照组
    # # This section needs updating if control group analysis is still needed separately
    # print("\n==== Analyzing control group data (only subject 8) ====")
    # control_config_list = []
    # for config in [nback_acc_config, nback_rt_config]: # Need to create configs for control group
    #     # 创建新配置，避免修改原始配置
    #     control_config = config.copy()
    #     control_config["plot_config"] = config["plot_config"].copy()

    #     # 更新输出文件名，加上control前缀
    #     output_file = config["plot_config"]["output_file"]
    #     control_config["plot_config"]["output_file"] = output_file.replace(
    #         ".png", "_control.png"
    #     )
    #     control_config["plot_config"]["title"] = control_config["plot_config"]["title"].replace(
    #         "Comparison", "Comparison (Control Group)" # Modify title
    #     )

    #     control_config_list.append(control_config)

    # print("Starting analysis for control group...")
    # analyze_experiment_accuracy(control_config_list[0], only_participant=8)
    # analyze_nback_rt(control_config_list[1], only_participant=8)
    # print("Control group analysis completed!")
