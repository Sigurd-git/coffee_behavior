import numpy as np
from psychopy import visual, core, event
import pandas as pd
from utils.common import show_instructions, save_data
from utils.config import (
    FEEDBACK_DURATION,
    INSTRUCTION_DURATION,
    MIN_ISI,
    MAX_ISI,
    RESPONSE_KEYS,
)


def run_go_no_go(win, participant_id, session):
    """
    Go/No-Go task implementation

    Parameters:
    - win: PsychoPy window object
    - participant_id: Participant identifier
    - session: Session number (1: Pre-coffee, 2: Post-coffee)
    """
    # Task parameters
    params = {
        "num_trials": 100,
        "go_probability": 0.8,  # 80% Go trials
        "stim_duration": 2,
        "isi": [MIN_ISI, 1.5, MAX_ISI],  # Variable ISI
        "practice_trials": 10,
        "response_window": 2.0,
        "feedback_duration": FEEDBACK_DURATION,
    }

    # Save experiment parameters
    exp_info = {
        "participant_id": participant_id,
        "session": session,
        "task": "Go/No-Go",
        "parameters": params,
    }

    # Create stimuli with standardized size
    circle_radius = win.size[1] / 10  # Relative to screen height
    go_stim = visual.Circle(
        win=win, radius=circle_radius, fillColor="green", lineColor="green"
    )
    no_go_stim = visual.Circle(
        win=win, radius=circle_radius, fillColor="red", lineColor="red"
    )

    # Show instructions
    instructions = """
    In this task:
    Press SPACE when you see a GREEN circle (Go)
    Do NOT press any key when you see a RED circle (No-Go)
    
    Try to respond as quickly and accurately as possible.
    
    Press any key to start practice.
    """
    show_instructions(win, instructions)

    # Initialize data recording
    data = {
        "trial": [],
        "trial_type": [],  # 'go' or 'no-go'
        "response": [],  # True if response made
        "rt": [],  # Response time
        "correct": [],  # Whether response was correct
    }

    def run_trial_block(num_trials, is_practice=False):
        """Run a block of trials"""
        for trial in range(num_trials):
            # Determine trial type
            is_go_trial = np.random.random() < params["go_probability"]

            # Show fixation
            fixation = visual.TextStim(win, text="+", height=win.size[1] / 20)
            fixation.draw()
            win.flip()
            core.wait(np.random.choice(params["isi"]))

            # Show stimulus
            if is_go_trial:
                go_stim.draw()
            else:
                no_go_stim.draw()

            win.flip()
            trial_clock = core.Clock()

            # Get response
            keys = event.waitKeys(
                maxWait=params["stim_duration"], keyList=[RESPONSE_KEYS["confirm"]]
            )
            rt = trial_clock.getTime() if keys else None
            response_made = bool(keys)

            # Determine if response was correct
            correct = response_made == is_go_trial

            if not is_practice:
                # Record data
                data["trial"].append(trial)
                data["trial_type"].append("go" if is_go_trial else "no-go")
                data["response"].append(response_made)
                data["rt"].append(rt)
                data["correct"].append(correct)

            # Clear screen
            win.flip()

            # Show feedback during practice
            if is_practice:
                feedback = "Correct!" if correct else "Incorrect!"
                feedback_stim = visual.TextStim(
                    win, text=feedback, height=win.size[1] / 20
                )
                feedback_stim.draw()
                win.flip()
                core.wait(params["feedback_duration"])

    # Run practice trials
    practice_text = visual.TextStim(
        win=win, text="Practice phase starting...", height=win.size[1] / 20
    )
    practice_text.draw()
    win.flip()
    core.wait(INSTRUCTION_DURATION)

    run_trial_block(params["practice_trials"], is_practice=True)

    # Start main experiment
    exp_text = visual.TextStim(
        win=win,
        text="Main experiment starting...\n\nPress any key to continue",
        height=win.size[1] / 20,
    )
    exp_text.draw()
    win.flip()
    event.waitKeys()

    # Run main trials
    run_trial_block(params["num_trials"], is_practice=False)

    # Save data
    filename = f"data/gng_{participant_id}_session{session}.csv"
    df = pd.DataFrame(data)
    df["participant_id"] = participant_id
    df["session"] = session
    df["task"] = "Go/No-Go"
    save_data(df, filename)

    # Calculate and display summary statistics
    go_trials = df[df["trial_type"] == "go"]
    no_go_trials = df[df["trial_type"] == "no-go"]

    summary = {
        "Go Accuracy": go_trials["correct"].mean() * 100,
        "No-Go Accuracy": no_go_trials["correct"].mean() * 100,
        "Mean Go RT": go_trials["rt"].mean() * 1000,  # Convert to ms
        "Commission Errors": (1 - no_go_trials["correct"].mean()) * 100,
        "Omission Errors": (1 - go_trials["correct"].mean()) * 100,
    }

    print("\nTask Summary:")
    for key, value in summary.items():
        print(f"{key}: {value:.2f}")
