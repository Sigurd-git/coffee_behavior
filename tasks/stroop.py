import numpy as np
from psychopy import visual, core, event
import pandas as pd
from utils.common import show_instructions, save_data


def run_stroop(win, participant_id, session):
    """
    Stroop task implementation

    Parameters:
    - win: PsychoPy window object
    - participant_id: Participant identifier
    - session: Session number (1: Pre-coffee, 2: Post-coffee)
    """
    # Task parameters (moved to top for better organization)
    params = {
        "num_trials": 8,
        "num_stimuli_per_trial": 12,
        "stim_duration": 0.5,
        "isi": 1.0,
        "num_practice_stimuli": 2,
        "colors": {"Red": [1, -1, -1], "Green": [-1, 1, -1], "Blue": [-1, -1, 1]},
        "words": ["红色", "绿色", "蓝色"],
        "neutral_word": "白色",
    }

    # # test
    # params = {
    #     "num_trials": 1,
    #     "num_stimuli_per_trial": 2,
    #     "stim_duration": 0.5,
    #     "isi": 1.0,
    #     "num_practice_stimuli": 2,
    #     "colors": {"Red": [1, -1, -1], "Green": [-1, 1, -1], "Blue": [-1, -1, 1]},
    #     "words": ["红色", "绿色", "蓝色"],
    #     "neutral_word": "白色",
    # }

    # Save experiment parameters
    exp_info = {
        "participant_id": participant_id,
        "session": session,
        "task": "Stroop",
        "parameters": params,
    }

    # Create stimuli with standardized size
    text_stim = visual.TextStim(
        win=win,
        height=win.size[1] / 10,  # Standardized size relative to screen
        font="SimHei",
        wrapWidth=None,
        color="white",
    )

    # Instructions
    instructions = """
    In this task, ignore the meaning of the words and respond to their colors.

    Press:
    1 for Red
    2 for Green
    3 for Blue

    Press any key to begin
    """
    show_instructions(win, instructions)

    # Initialize data recording
    data = {
        "trial": [],
        "word": [],
        "color": [],
        "response": [],
        "rt": [],
        "correct": [],
    }

    def run_trial_block(is_practice=False):
        """Run a block of trials"""
        for trial in range(
            params["num_stimuli_per_trial"]
            if is_practice
            else params["num_trials"] * params["num_stimuli_per_trial"]
        ):
            # Generate random stimulus
            word = np.random.choice(params["words"])
            color_name = np.random.choice(list(params["colors"].keys()))

            # Present stimulus
            text_stim.text = word
            text_stim.color = params["colors"][color_name]
            text_stim.draw()

            win.flip()
            core.wait(params["stim_duration"])

            # Get response
            clock = core.Clock()
            keys = event.waitKeys(maxWait=2.0, keyList=["1", "2", "3"])

            if keys:
                rt = clock.getTime()
                response = int(keys[0])
                correct = (
                    response == list(params["colors"].keys()).index(color_name) + 1
                )

                if not is_practice:
                    # Record trial data
                    data["trial"].append(trial)
                    data["word"].append(word)
                    data["color"].append(color_name)
                    data["response"].append(response)
                    data["rt"].append(rt)
                    data["correct"].append(correct)

                # Show feedback during practice
                if is_practice:
                    feedback = "Correct!" if correct else "Incorrect!"
                    text_stim.text = feedback
                    text_stim.color = "white"
                    text_stim.draw()
                    win.flip()
                    core.wait(0.5)

            win.flip()
            core.wait(params["isi"])

    # Run practice trials
    practice_text = visual.TextStim(
        win=win, text="Practice phase starting...", height=30
    )
    practice_text.draw()
    win.flip()
    core.wait(2)

    run_trial_block(is_practice=True)

    # Start main experiment
    exp_text = visual.TextStim(
        win=win,
        text="Main experiment starting...\n\nPress any key to continue",
        height=30,
    )
    exp_text.draw()
    win.flip()
    event.waitKeys()

    # Run main trials
    run_trial_block(is_practice=False)

    # Save data
    filename = f"data/stroop_{participant_id}_session{session}.csv"
    save_data(data, filename)

    # Calculate and display summary statistics
    df = pd.DataFrame(data)
    summary = {
        "Accuracy": df["correct"].mean() * 100,
        "Mean RT": df["rt"].mean() * 1000,  # Convert to ms
        "Congruent Accuracy": df[df["word"] == df["color"]]["correct"].mean() * 100,
        "Incongruent Accuracy": df[df["word"] != df["color"]]["correct"].mean() * 100,
    }

    print("\nTask Summary:")
    for key, value in summary.items():
        print(f"{key}: {value:.2f}")
