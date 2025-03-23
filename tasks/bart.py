import numpy as np
from psychopy import visual, core, event
import pandas as pd
from utils.common import show_instructions, save_data
from tasks.config import *


def run_bart(win, participant_id, session):
    """
    Balloon Analogue Risk Task (BART) implementation

    Parameters:
    - win: PsychoPy window object
    - participant_id: Participant identifier
    - session: Session number (1: Pre-coffee, 2: Post-coffee)
    
    """
    # Task parameters
    params = {
        "num_balloons": 10,
        "max_pumps": 11,
        "practice_balloons": 1,
        "reward_per_pump": 5,  # ¥5 per pump
        "initial_radius": 50,
        "pump_increment": 10,
        "max_radius": 300,
        "explosion_increment": 0.05,  # 5% increase per pump
        "feedback_duration": FEEDBACK_DURATION,
    }

    # Save experiment parameters
    exp_info = {
        "participant_id": participant_id,
        "session": session,
        "task": "BART",
        "parameters": params,
    }

    # Create stimuli with standardized size
    balloon = visual.Circle(
        win=win,
        radius=params["initial_radius"],
        fillColor="blue",
        lineColor="white",
        lineWidth=3,
    )

    # Create text display with standardized size
    text_display = visual.TextStim(
        win=win,
        height=win.size[1] / 25,  # Reduced text size (from 1/20 to 1/25)
        color="white",
        pos=(0, -win.size[1] / 4),  # Moved up (from 1/3 to 1/4)
    )

    # Show instructions
    instructions = """
    气球充气游戏：
    在这个环节中，你所获得的金钱的5%将真实加入你的被试费！
    调整气球大小，赚取更多金钱！运气好的话，奖金最高可达200元人民币！
    - 按【空格键】给气球充气
    - 按【回车键】收集已有金钱
    - 小心！气球可能会爆炸！爆炸了则失去已有金钱！
    - 气球越大 = 赚得越多，但风险也越大！
    （在屏幕下方会显示当前已获得的金钱和气球爆炸概率，
      爆炸概率随充气次数增加而增加）
    
    按任意键开始练习。第一个气球是练习试次，以了解规则，不计入被试费。
    """
    show_instructions(win, instructions)

    # Initialize data recording
    data = {
        "balloon": [],
        "pumps": [],
        "exploded": [],
        "earned": [],
        "rt": [],
    }

    def run_trial_block(num_trials, is_practice=False):
        """Run a block of BART trials"""
        total_earned = 0

        for trial in range(num_trials):
            pumps = 0
            exploded = False
            trial_clock = core.Clock()

            while True:
                # Update balloon size
                balloon.radius = params["initial_radius"] + (
                    pumps * params["pump_increment"]
                )
                balloon.draw()

                # Show trial info
                text_display.text = (
                    f"气球 #{trial + 1}\n"
                    f"当前充气次数: {pumps}\n"
                    f"当前价值: ¥{pumps * params['reward_per_pump']:.2f}\n"
                    f"累计收益: ¥{total_earned:.2f}\n"
                    f"当前爆炸概率: {pumps * params['explosion_increment']*100:.1f}%"
                )
                text_display.draw()

                win.flip()

                # Get response
                keys = event.waitKeys(keyList=["space", "return", "escape"])

                if "escape" in keys:
                    win.close()
                    core.quit()

                if "return" in keys:  # Collect money
                    break

                if "space" in keys:  # Pump balloon
                    pumps += 1
                    # Check for explosion
                    explosion_probability = (pumps - 1) * params["explosion_increment"]
                    if np.random.random() < explosion_probability:
                        exploded = True
                        break

                    if pumps >= params["max_pumps"]:
                        break

            # Trial outcome
            if not exploded:
                earned = pumps * params["reward_per_pump"]
                total_earned += earned
                # Show outcome
                text_display.text = f"你赚到了 ¥{earned:.2f}！恭喜！"
                text_display.draw()
                win.flip()
                core.wait(params["feedback_duration"])
            else:
                earned = 0
                # Create explosion feedback text
                explosion_text = visual.TextStim(
                    win=win,
                    text="气球爆炸啦！\n请再接再厉！\n按空格继续",
                    height=win.size[1] / 20,
                    color="white",
                    bold=True
                )
                # Display explosion feedback
                explosion_text.draw()
                win.flip()
                # Wait for spacebar press
                event.waitKeys(keyList=['space'])

            if not is_practice:
                # Record data
                data["balloon"].append(trial)
                data["pumps"].append(pumps)
                data["exploded"].append(exploded)
                data["earned"].append(earned)
                data["rt"].append(trial_clock.getTime())

        return total_earned

    # Run practice
    practice_text = visual.TextStim(
        win=win, text="练习阶段", height=win.size[1] / 20
    )
    practice_text.draw()
    win.flip()
    core.wait(INSTRUCTION_DURATION)

    run_trial_block(params["practice_balloons"], is_practice=True)

    # Start main experiment
    exp_text = visual.TextStim(
        win=win,
        text="正式实验即将开始...\n\n按任意键继续",
        height=win.size[1] / 20,
    )
    exp_text.draw()
    win.flip()
    event.waitKeys()

    # Run main trials
    total_earned = run_trial_block(params["num_balloons"], is_practice=False)

    # Save data
    filename = f"data/bart_{participant_id}_session{session}.csv"
    df = pd.DataFrame(data)
    df["participant_id"] = participant_id
    df["session"] = session
    df["task"] = "BART"
    save_data(df, filename)

    # Calculate and display summary statistics
    summary = {
        "平均充气次数": df["pumps"].mean(),
        "爆炸率": df["exploded"].mean() * 100,
        "总收益": total_earned,
        "每个气球平均收益": df["earned"].mean(),
    }

    # 创建summary DataFrame
    summary_df = pd.DataFrame([["", "", "", "", "", "", "", ""], 
                             ["Summary Statistics", "", "", "", "", "", "", ""],
                             ["平均充气次数", summary["平均充气次数"], "", "", "", "", "", ""],
                             ["爆炸率(%)", summary["爆炸率"], "", "", "", "", "", ""],
                             ["总收益(¥)", summary["总收益"], "", "", "", "", "", ""],
                             ["每个气球平均收益(¥)", summary["每个气球平均收益"], "", "", "", "", "", ""]],
                             columns=df.columns)

    # 合并原始数据和summary
    final_df = pd.concat([df, summary_df], ignore_index=True)

    # 保存完整数据
    filename = f"data/bart_{participant_id}_session{session}.csv"
    save_data(final_df, filename)

    # 打印summary
    print("\n任务总结:")
    for key, value in summary.items():
        print(f"{key}: {value:.2f}")
