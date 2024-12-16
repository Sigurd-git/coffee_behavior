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
        height=win.size[1] / 20,  # Smaller text for info display
        color="white",
        pos=(0, -win.size[1] / 3),  # Position relative to screen height
    )

    # Show instructions
    instructions = """
    气球充气游戏：
    
    - 按空格键给气球充气
    - 按回车键收集现有金钱
    - 小心！气球可能会爆炸！
    - 气球越大 = 赚得越多，但风险也越大！
    
    按任意键开始练习。
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
                outcome_text = f"你赚到了 ¥{earned:.2f}！"
            else:
                earned = 0
                outcome_text = "气球爆炸了！没有收益。"

            if not is_practice:
                # Record data
                data["balloon"].append(trial)
                data["pumps"].append(pumps)
                data["exploded"].append(exploded)
                data["earned"].append(earned)
                data["rt"].append(trial_clock.getTime())

            # Show outcome
            text_display.text = outcome_text
            text_display.draw()
            win.flip()
            core.wait(params["feedback_duration"])

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

    print("\n任务总结:")
    for key, value in summary.items():
        print(f"{key}: {value:.2f}")
