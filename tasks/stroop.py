import numpy as np
from psychopy import visual, core, event
import pandas as pd
from utils.common import show_instructions, save_data


def run_stroop(win, participant_id, session):
    """
    Stroop任务实现

    参数:
    - win: PsychoPy窗口对象
    - participant_id: 被试编号
    - session: 实验阶段编号 (1: 喝咖啡前, 2: 喝咖啡后)
    """
    # 任务参数设置
    params = {
        "num_trials": 8,  # 每组试次数
        "num_stimuli_per_trial": 12,  # 每个试次的刺激数量
        "stim_duration": 0.5,  # 刺激呈现时间
        "isi": 1.0,  # 刺激间隔时间
        "num_practice_stimuli": 2,  # 练习试次数量
        "colors": {"Red": [1, -1, -1], "Green": [-1, 1, -1], "Blue": [-1, -1, 1]},  # 颜色RGB值
        "words": ["红", "绿", "蓝"],  # 文字刺激
        "neutral_word": "白",  # 中性词
    }
    # # develope 任务参数设置
    # params = {
    #     "num_trials": 1,  # 每组试次数
    #     "num_stimuli_per_trial": 2,  # 每个试次的刺激数量
    #     "stim_duration": 0.5,  # 刺激呈现时间
    #     "isi": 1.0,  # 刺激间隔时间
    #     "num_practice_stimuli": 1,  # 练习试次数量
    #     "colors": {"Red": [1, -1, -1], "Green": [-1, 1, -1], "Blue": [-1, -1, 1]},  # 颜色RGB值
    #     "words": ["红色", "绿色", "蓝色"],  # 文字刺激
    #     "neutral_word": "白色",  # 中性词
    # }

    # 保存实验参数
    exp_info = {
        "participant_id": participant_id,
        "session": session,
        "task": "Stroop",
        "parameters": params,
    }

    # 创建标准大小的刺激
    text_stim = visual.TextStim(
        win=win,
        height=win.size[1] / 10,  # 相对于屏幕高度的标准化大小
        font="SimHei",
        wrapWidth=None,
        color="white",
    )

    # 指导语
    instructions = """
    在这个任务中，请你忽略文字的含义，仅对文字的颜色做出反应。

    按键说明:
    1 键 = 红色
    2 键 = 绿色
    3 键 = 蓝色
    即红颜色的文字按1键，绿颜色的文字按2键，蓝颜色的文字按3键
    不必理会文字的含义。请快速而准确地做出反应。
    你可以稍作休息，当准备好了，按任意键开始
    """
    show_instructions(win, instructions)

    # 初始化数据记录
    data = {
        "trial": [],  # 试次编号
        "word": [],   # 呈现的文字
        "color": [],  # 呈现的颜色
        "response": [],  # 被试反应
        "rt": [],     # 反应时
        "correct": [],  # 是否正确
    }

    def run_trial_block(block_num, is_practice=False):
        """运行一组试次"""
        block_data = {
            "trial": [],
            "word": [],
            "color": [],
            "response": [],
            "rt": [],
            "correct": [],
            "block": []
        }

        # 所有可能的词（包括中性词"白色"）
        all_words = params["words"] + [params["neutral_word"]]  # ["红色", "绿色", "蓝色", "白色"]
        all_colors = list(params["colors"].keys())  # ["Red", "Green", "Blue"]

        # 运行试次
        for trial in range(params["num_stimuli_per_trial"]):  # 12个试次
            # 随机选择词和颜色
            word = np.random.choice(all_words)
            color_name = np.random.choice(all_colors)

            # 呈现刺激
            text_stim.text = word
            text_stim.color = params["colors"][color_name]
            text_stim.draw()

            win.flip()
            core.wait(params["stim_duration"])

            # 获取反应
            clock = core.Clock()
            keys = event.waitKeys(maxWait=2.0, keyList=["1", "2", "3"])

            if keys:
                rt = clock.getTime()
                response = int(keys[0])
                correct = (
                    response == list(params["colors"].keys()).index(color_name) + 1
                )

                if not is_practice:
                    # 记录试次数据
                    block_data["trial"].append(trial)
                    block_data["word"].append(word)
                    block_data["color"].append(color_name)
                    block_data["response"].append(response)
                    block_data["rt"].append(rt)
                    block_data["correct"].append(correct)
                    block_data["block"].append(block_num)

                # 练习阶段显示反馈
                if is_practice:
                    feedback = "正确！" if correct else "错误！"
                    text_stim.text = feedback
                    text_stim.color = "white"
                    text_stim.draw()
                    win.flip()
                    core.wait(0.5)

            win.flip()
            core.wait(params["isi"])
        
        return block_data

    # 运行练习试次
    practice_text = visual.TextStim(
        win=win, text="练习阶段开始\n\n按任意键继续\n练习阶段有反馈", height=30
    )
    practice_text.draw()
    win.flip()
    event.waitKeys()

    run_trial_block(0, is_practice=True)

    # 初始化所有数据存储
    all_data = []

    # 运行8个正式block
    for block in range(params["num_trials"]):
        # 显示block开始信息
        block_text = visual.TextStim(
            win=win,
            text=f"第 {block + 1} 组（共8组）\n\n你可以稍作休息\n\n准备好后按任意键继续",
            height=30,
        )
        block_text.draw()
        win.flip()
        event.waitKeys()

        # 运行当前block
        block_data = run_trial_block(block, is_practice=False)
        all_data.append(pd.DataFrame(block_data))

    # 合并所有block的数据
    df = pd.concat(all_data, ignore_index=True)

    # 在计算统计数据的部分修改
    def calculate_block_stats(block_df):
        """计算单个block的统计数据"""
        return {
            "正确率": block_df["correct"].mean() * 100,
            "平均反应时": block_df["rt"].mean() * 1000,
            "一致条件正确率": block_df[block_df["word"] == block_df["color"]]["correct"].mean() * 100,
            "不一致条件正确率": block_df[block_df["word"] != block_df["color"]]["correct"].mean() * 100,
            "中性词正确率": block_df[block_df["word"] == "白色"]["correct"].mean() * 100,
            "一致条件反应时": block_df[block_df["word"] == block_df["color"]]["rt"].mean() * 1000,
            "不一致条件反应时": block_df[block_df["word"] != block_df["color"]]["rt"].mean() * 1000,
            "中性词反应时": block_df[block_df["word"] == "白色"]["rt"].mean() * 1000
        }

    # 计算每个block和总体的统计数据
    stats = []

    # 计算每个block的统计
    for block in range(params["num_trials"]):
        block_df = df[df["block"] == block]
        block_stats = calculate_block_stats(block_df)
        stats.append({
            "level": "block",
            "block": block + 1,
            **block_stats,
            "participant_id": participant_id,
            "session": session
        })

    # 计算总体统计
    total_stats = calculate_block_stats(df)
    stats.append({
        "level": "total",
        "block": "all",
        **total_stats,
        "participant_id": participant_id,
        "session": session
    })

    # 创建统计数据DataFrame
    stats_df = pd.DataFrame(stats)

    # 保存原始数据和统计数据
    raw_filename = f"data/stroop_{participant_id}_session{session}.csv"
    save_data(df, raw_filename)

    stats_filename = f"data/stroop_stats_{participant_id}_session{session}.csv"
    save_data(stats_df, stats_filename)

    # 打印统计结果
    print("\n任务总结:")
    print("\n每个Block的统计:")
    block_stats = stats_df[stats_df["level"] == "block"]
    for _, row in block_stats.iterrows():
        print(f"\nBlock {row['block']}:")
        for key in ["正确率", "平均反应时", 
                    "一致条件正确率", "不一致条件正确率", "中性词正确率",
                    "一致条件反应时", "不一致条件反应时", "中性词反应时"]:
            print(f"  {key}: {row[key]:.2f}")

    print("\n总体统计:")
    total_stats = stats_df[stats_df["level"] == "total"].iloc[0]
    for key in ["正确率", "平均反应时", 
                "一致条件正确率", "不一致条件正确率", "中性词正确率",
                "一致条件反应时", "不一致条件反应时", "中性词反应时"]:
        print(f"  {key}: {total_stats[key]:.2f}")
