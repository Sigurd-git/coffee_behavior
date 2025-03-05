import numpy as np
from psychopy import visual, core, event
import pandas as pd
from utils.common import show_instructions, save_data
from tasks.config import *


def run_n_back(win, participant_id, session):
    """
    N-back task implementation

    Parameters:
    - win: PsychoPy window object
    - participant_id: Participant identifier
    - session: Session number (1: Pre-coffee, 2: Post-coffee)
    """
    # Task parameters
    params = {
        "n_back_levels": [0, 1, 2],
        "trials_per_block": 30,
        "num_blocks": 2,
        "stim_duration": 0.5,
        "isi": 1,
        "practice_trials": 10,
        "target_probability": 0.25,
        "letters": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "feedback_duration": FEEDBACK_DURATION,
    }
    # # test params
    # params = {
    #     "n_back_levels": [0, 1, 2],
    #     "trials_per_block": 2,
    #     "num_blocks": 2,
    #     "stim_duration": 0.5,
    #     "isi": 1.5,
    #     "practice_trials": 1,
    #     "target_probability": 0.25,
    #     "letters": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    #     "feedback_duration": FEEDBACK_DURATION,
    # }

    # Save experiment parameters
    exp_info = {
        "participant_id": participant_id,
        "session": session,
        "task": "N-back",
        "parameters": params,
    }

    # Create stimuli with standardized size
    text_stim = visual.TextStim(win=win, height=win.size[1] / 8, color="white")

    def show_n_back_instructions(n):
        """显示n-back任务的具体指导语"""
        if n == 0:
            inst_text = """
            欢迎来到n-back任务！我们将面对一系列的字母

            0-back 任务：
            在接下来的字母串中，
            当你看到字母 'X' 时，请按空格键
            反应时间较短，请尽可能快速且准确地做出反应。
            
            你可以稍作休息，当准备好了，按任意键开始练习试次，
            练习阶段有正误反馈，反应太慢也会算作错误
            """
        else:
            inst_text = f"""
            {n}-back 任务：
            在接下来的字母串中
            当出现的字母与上{n}个位置的字母相同时，按空格键
            反应时间较短，请尽可能快速且准确地做出反应。
            （如，接连呈现X M N，
            X是N的上两个字母，M是N的上一个字母
            我们需要记住这些过去的字母，并和新出现的字母进行对比）
            请保证已理解任务规则

            你可以稍作休息，当准备好了，按任意键开始练习，
            练习阶段有正误反馈，反应太慢也会算作错误
            """
        show_instructions(win, inst_text)

    def generate_sequence(n, num_trials):
        """Generate sequence of stimuli with targets"""
        sequence = []
        targets = []

        if n == 0:
            target_letter = "X"
            for _ in range(num_trials):
                if np.random.random() < params["target_probability"]:
                    sequence.append(target_letter)
                    targets.append(True)
                else:
                    sequence.append(
                        np.random.choice(
                            [l for l in params["letters"] if l != target_letter]
                        )
                    )
                    targets.append(False)
        else:
            for i in range(num_trials):
                if i < n:
                    sequence.append(np.random.choice(params["letters"]))
                    targets.append(False)
                else:
                    if np.random.random() < params["target_probability"]:
                        sequence.append(sequence[i - n])
                        targets.append(True)
                    else:
                        sequence.append(
                            np.random.choice(
                                [l for l in params["letters"] if l != sequence[i - n]]
                            )
                        )
                        targets.append(False)

        return sequence, targets

    def generate_practice_sequence(n, sequence_length=10):
        """
        Generate a practice sequence that guarantees target trials
        
        Parameters:
        - n: n-back value
        - sequence_length: length of the sequence
        
        Returns:
        - sequence: list of stimuli
        - targets: list of boolean values indicating target positions
        """
        stimuli = ['A', 'B', 'C', 'D']
        sequence = []
        targets = []
        
        # Ensure at least 2 targets in the sequence
        min_targets = 2
        target_positions = np.random.choice(
            range(n, sequence_length), 
            size=min_targets, 
            replace=False
        )
        
        # Generate initial n items
        for i in range(n):
            sequence.append(np.random.choice(stimuli))
            targets.append(False)
        
        # Generate remaining items
        for i in range(n, sequence_length):
            if i in target_positions:
                # Create a target by using the same letter as n positions back
                sequence.append(sequence[i-n])
                targets.append(True)
            else:
                # Add a random non-target letter
                prev_letter = sequence[i-n]
                available_letters = [x for x in stimuli if x != prev_letter]
                sequence.append(np.random.choice(available_letters))
                targets.append(False)
        
        return sequence, targets

    def run_block(n, num_trials, is_practice=False):
        """运行一个n-back任务区组"""
        sequence, targets = generate_sequence(n, num_trials)

        block_data = {
            "trial": [],
            "stimulus": [],
            "target": [],
            "response": [],
            "rt": [],
            "correct": [],
        }

        # Show fixation at the beginning of block
        text_stim.pos = (0, 0)
        text_stim.text = "+"
        text_stim.draw()
        win.flip()
        core.wait(0.5)

        # Run trials
        for i, (stim, is_target) in enumerate(zip(sequence, targets)):
            # 为字母添加随机微小位移
            random_offset = 20  # 像素偏移范围
            x_offset = np.random.uniform(-random_offset, random_offset)
            y_offset = np.random.uniform(-random_offset, random_offset)
            text_stim.pos = (x_offset, y_offset)
            
            # Show stimulus
            text_stim.text = stim
            text_stim.draw()
            win.flip()

            # Get response
            trial_clock = core.Clock()
            keys = event.waitKeys(
                maxWait=params["stim_duration"],
                keyList=[RESPONSE_KEYS["confirm"]],
            )
            rt = trial_clock.getTime() if keys else None
            response_made = bool(keys)

            # Determine if response was correct
            correct = response_made == is_target

            if not is_practice:
                # Record data
                block_data["trial"].append(i)
                block_data["stimulus"].append(stim)
                block_data["target"].append(is_target)
                block_data["response"].append(response_made)
                block_data["rt"].append(rt)
                block_data["correct"].append(correct)

            # Clear screen
            win.flip()

            # Show feedback during practice
            if is_practice and i >= n:
                show_feedback = False
                feedback = ""
                
                print(f"Debug - Target: {is_target}, Response: {response_made}")  # 添加调试信息
                
                if is_target and not response_made:  # 需要按但没按
                    show_feedback = True
                    feedback = "错误！"
                elif is_target and response_made:  # 需要按而且按了
                    show_feedback = True
                    feedback = "正确！"
                elif not is_target and response_made:  # 不需要按但按了
                    show_feedback = True
                    feedback = "错误！"
                # 不需要按且没按的情况不显示反馈
                
                if show_feedback:
                    text_stim.height = win.size[1] / 25
                    text_stim.text = feedback
                    text_stim.draw()
                    win.flip()
                    core.wait(params["feedback_duration"])
                    text_stim.height = win.size[1] / 8

            # ISI
            core.wait(params["isi"])

        return block_data

    # Initialize overall data storage
    all_data = []

    # Run blocks for each n-back level
    for n in params["n_back_levels"]:
        # Show instructions
        show_n_back_instructions(n)

        # Practice block
        practice_text = visual.TextStim(
            win=win, text=f"{n}-back 练习阶段", height=win.size[1] / 20
        )
        practice_text.draw()
        win.flip()
        core.wait(INSTRUCTION_DURATION)

        if is_practice:
            sequence, target_positions = generate_practice_sequence(n)
        else:
            # 保持原有的实验序列生成逻辑不变
            ...

        run_block(n, params["practice_trials"], is_practice=True)

        # Main blocks
        for block in range(params["num_blocks"]):
            block_text = visual.TextStim(
                win=win,
                text=f"{n}-back 第{block+1}组\n正式实验阶段没有正误反馈。\n你可以稍作休息，\n当准备好了，\n继续进行字母比较，\n按任意键开始",
                height=win.size[1] / 25,  # Reduced text size from 1/20 to 1/25
            )
            block_text.draw()
            win.flip()
            event.waitKeys()

            block_data = run_block(n, params["trials_per_block"])
            block_data = pd.DataFrame(block_data)
            block_data["n_back"] = n
            block_data["block"] = block
            block_data["participant_id"] = participant_id
            block_data["session"] = session
            block_data["task"] = "N-back"
            all_data.append(block_data)

    # 合并所有数据
    df = pd.concat(all_data, ignore_index=True)

    # 计算统计数据
    stats_df = calculate_nback_stats(df)

    # 保存原始数据
    raw_filename = f"data/nback_{participant_id}_session{session}.csv"
    save_data(df, raw_filename)

    # 保存统计数据
    stats_filename = f"data/nback_stats_{participant_id}_session{session}.csv"
    save_data(stats_df, stats_filename)

    # Calculate and display summary statistics
    summary = (
        df.groupby("n_back")
        .agg(
            {
                "correct": ["mean", "std"],
                "rt": lambda x: np.mean([r for r in x if r is not None]),
            }
        )
        .round(3)
    )

    print("\nTask Summary:")
    print(summary)

def calculate_nback_stats(df):
    """计算每个block和每个n-back水平的平均反应时和正确率"""
    stats = []
    
    # 计算每个block的统计数据
    block_stats = df.groupby(['n_back', 'block']).agg({
        'correct': 'mean',  # 计算平均正确率
        'rt': lambda x: np.mean([r for r in x if r is not None])  # 平均反应时（排除None）
    }).reset_index()
    
    # 添加block级别的统计
    for _, row in block_stats.iterrows():
        stats.append({
            'level': 'block',
            'n_back': row['n_back'],
            'block': row['block'],
            'accuracy': row['correct'] * 100,  # 转换为百分比
            'mean_rt': row['rt'],
            'participant_id': df['participant_id'].iloc[0],
            'session': df['session'].iloc[0]
        })
    
    # 计算每个n-back水平的总体统计
    task_stats = df.groupby('n_back').agg({
        'correct': 'mean',
        'rt': lambda x: np.mean([r for r in x if r is not None])
    }).reset_index()
    
    # 添加task级别的统计
    for _, row in task_stats.iterrows():
        stats.append({
            'level': 'task',
            'n_back': row['n_back'],
            'block': 'all',  # 表示所有block的平均
            'accuracy': row['correct'] * 100,  # 转换为百分比
            'mean_rt': row['rt'],
            'participant_id': df['participant_id'].iloc[0],
            'session': df['session'].iloc[0]
        })
    
    return pd.DataFrame(stats)
