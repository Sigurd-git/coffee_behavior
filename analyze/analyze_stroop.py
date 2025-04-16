import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.plot import plot_bar_comparison
from utils.tasks import extract_stroop
from statannotations.Annotator import Annotator
import statsmodels.api as sm
from statsmodels.formula.api import ols

def analyze_ies(experiment_config, exclude_participant=None, only_participant=None):
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
    print("\n==== ANOVA IES ====")
    model = ols("ies ~ condition * session * group", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    # 6. 可视化
    if experiment_config.get("plot", True):
        plot_config = experiment_config.get("plot_config", {})
        row_var = plot_config.get("row_var", None)
        if row_var is not None:
            row_values = df[row_var].unique()
            fig, axes = plt.subplots(
                figsize=plot_config.get("figsize", (8, 5)),
                nrows=len(row_values),
                ncols=1,
            )
        else:
            row_values = [None]
            fig, ax = plt.subplots(figsize=plot_config.get("figsize", (8, 5)))
            axes = [ax]
        title = plot_config.get(
            "title",
            f"{experiment_config['experiment_type']} Task Accuracy Comparison",
        )
        fig.suptitle(title)
        for row_value, ax in zip(row_values, axes):
            if row_value is not None:
                df_row = df[df[row_var] == row_value]
            else:
                df_row = df
            plot_bar_comparison(
                df_row,
                x_var=plot_config.get("x_var", "condition"),
                y_var=plot_config.get("y_var", "accuracy"),  # 使用accuracy作为y轴变量
                hue_var=plot_config.get("hue_var", "session"),
                xlabel=plot_config.get("xlabel", "Condition"),
                ylabel=plot_config.get("ylabel", "Accuracy (%)"),
                palette=plot_config.get("palette", ["lightblue", "lightgreen"]),
                ax=ax,
                show_values=plot_config.get("show_values", True),
            )
            ax.set_title(row_value)
            # 添加显著性标记
            x_var = plot_config.get("x_var", "condition")
            y_var = plot_config.get("y_var", "accuracy")  # 使用正确的y变量名
            hue_var = plot_config.get("hue_var", "session")
            x_values = df_row[x_var].unique()
            hue_values = df_row[hue_var].unique()
            box_pairs = []
            for x in x_values:
                for i in range(len(hue_values)):
                    for j in range(i + 1, len(hue_values)):
                        box_pairs.append(((x, hue_values[i]), (x, hue_values[j])))
            annotator = Annotator(
                ax,
                box_pairs,
                data=df_row,
                x=x_var,
                y=y_var,
                order=x_values,
                hue=hue_var,
            )
            annotator.configure(
                test="t-test_paired",
                text_format="star",
                loc="inside",
                hide_non_significant=True,
            )
            annotator.apply_and_annotate()
        plt.tight_layout()

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
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    return results


# 示例用法
if __name__ == "__main__":
    # Stroop实验配置
    stroop_acc_config = {
        "experiment_type": "stroop",
        "extract_function": extract_stroop,
        "plot_config": {
            "x_var": "condition",
            "y_var": "ies",
            "row_var": "group",
            "title": "Stroop Task IES Comparison",
            "xlabel": "Session",
            "ylabel": "IES",
            "output_file": "stroop_ies_comparison.png",
        },
        "output_dir": "output",
    }

    # 创建输出目录
    os.makedirs("output", exist_ok=True)

    results = analyze_ies(
        stroop_acc_config,
        exclude_participant=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 19, 16, 25],
    )

    print(
        "\nStarting analysis of individual experiment accuracy for experimental group..."
    )
    print("\nAccuracy analysis completed!")
