import numpy as np
from .stroop import run_stroop
from .go_no_go import run_go_no_go
from .n_back import run_n_back
from .bart import run_bart
from .emotion import run_emotion
import os


def run_experiment(win, participant_id, session):
    """
    Main control script for the experimental setup

    Parameters:
    - win: PsychoPy window object
    - participant_id: Participant identifier
    - session: Session number (1: Pre-coffee, 2: Post-coffee)
    """
    # Define tasks and Latin square order
    tasks = {
        # "Stroop": run_stroop,
        # "GoNoGo": run_go_no_go,
        "NBack": run_n_back,
        # "BART": run_bart,
        # "Emotion": run_emotion,
    }
    num_tasks = len(tasks)
    os.makedirs(f"data", exist_ok=True)
    # Generate Latin square order
    latin_square = np.zeros((num_tasks, num_tasks), dtype=int)
    for i in range(num_tasks):
        latin_square[i] = np.roll(np.arange(num_tasks), -i)

    # Assign specific order based on participant ID
    participant_order_index = int(participant_id) % num_tasks
    task_order = latin_square[participant_order_index]

    # Display task order for debugging
    task_names = list(tasks.keys())
    ordered_tasks = [task_names[i] for i in task_order]
    print(f"Task order for participant {participant_id}: {', '.join(ordered_tasks)}")

    # Run tasks in determined order
    for task_idx in task_order:
        task_name = task_names[task_idx]
        print(f"Starting task: {task_name}")
        task_func = tasks[task_name]
        task_func(win, participant_id, session)

    print("All tasks completed!")
