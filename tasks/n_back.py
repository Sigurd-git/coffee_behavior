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
        "isi": 1.5,
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
    text_stim = visual.TextStim(win=win, height=win.size[1] / 5, color="white")

    def show_n_back_instructions(n):
        """Show instructions for specific n-back level"""
        if n == 0:
            inst_text = """
            0-back task:
            Press SPACE when you see the letter 'X'
            
            Press any key to begin
            """
        else:
            inst_text = f"""
            {n}-back task:
            Press SPACE when the current letter matches
            the letter shown {n} position{'s' if n > 1 else ''} ago
            
            Press any key to begin
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

    def run_block(n, num_trials, is_practice=False):
        """Run a block of n-back trials"""
        sequence, targets = generate_sequence(n, num_trials)

        block_data = {
            "trial": [],
            "stimulus": [],
            "target": [],
            "response": [],
            "rt": [],
            "correct": [],
        }

        # Run trials
        for i, (stim, is_target) in enumerate(zip(sequence, targets)):
            # Show fixation
            text_stim.text = "+"
            text_stim.draw()
            win.flip()
            core.wait(0.5)

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
                feedback = "Correct!" if correct else "Incorrect!"
                text_stim.text = feedback
                text_stim.draw()
                win.flip()
                core.wait(params["feedback_duration"])

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
            win=win, text=f"{n}-back Practice Phase", height=win.size[1] / 20
        )
        practice_text.draw()
        win.flip()
        core.wait(INSTRUCTION_DURATION)

        run_block(n, params["practice_trials"], is_practice=True)

        # Main blocks
        for block in range(params["num_blocks"]):
            block_text = visual.TextStim(
                win=win,
                text=f"{n}-back Block {block+1}\n\nPress any key to begin",
                height=win.size[1] / 20,
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

    # Combine all data and save
    df = pd.concat(all_data, ignore_index=True)
    filename = f"data/nback_{participant_id}_session{session}.csv"
    save_data(df, filename)

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
