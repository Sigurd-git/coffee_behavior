import numpy as np
from psychopy import visual, core, event
import pandas as pd
from utils.common import show_instructions, save_data
from tasks.config import *
import os
from glob import glob

def run_emotion(win, participant_id, session):
    """
    Emotion Recognition Task implementation
    
    Parameters:
    - win: PsychoPy window object
    - participant_id: Participant identifier
    - session: Session number (1: Pre-coffee, 2: Post-coffee)
    """
    # Task parameters
    params = {
        'num_trials': 60,
        'stim_duration': 0.5,
        'response_window': 3.0,
        'isi': 1.0,
        'practice_trials': 6,
        'emotions': ["happy", "sad", "angry", "fearful", "neutral", "surprised"],
        'stim_size': (400, 400),
        'feedback_duration': FEEDBACK_DURATION,
        'stim_dir': "stimuli/emotions"
    }

    # Save experiment parameters
    exp_info = {
        'participant_id': participant_id,
        'session': session,
        'task': 'Emotion',
        'parameters': params
    }

    # Create mapping for emotion responses
    emotion_keys = {str(i+1): emotion for i, emotion in enumerate(params['emotions'])}

    # Create stimuli with standardized size
    image_stim = visual.ImageStim(
        win=win,
        size=params['stim_size'],
        units="pix"
    )

    # Show instructions
    instructions = """
    Emotion Recognition Task:
    
    You will see faces showing different emotions.
    Respond as quickly as possible with the following keys:
    
    1 = Happy
    2 = Sad
    3 = Angry
    4 = Fearful
    5 = Neutral
    6 = Surprised
    
    Press any key to start practice.
    """
    show_instructions(win, instructions)

    # Initialize data recording
    data = {
        "trial": [],
        "image": [],
        "true_emotion": [],
        "response": [],
        "correct": [],
        "rt": [],
    }

    # Load emotion stimuli files
    image_files = []
    image_emotions = []
    
    for emotion in params['emotions']:
        files = glob(os.path.join(params['stim_dir'], f"{emotion}*.jpg"))
        image_files.extend(files)
        image_emotions.extend([emotion] * len(files))

    def run_trial_block(num_trials, is_practice=False):
        """Run a block of emotion recognition trials"""
        # Select random subset of images
        trial_indices = np.random.choice(len(image_files), num_trials, replace=False)
        
        for trial, idx in enumerate(trial_indices):
            # Load and present image
            image_path = image_files[idx]
            true_emotion = image_emotions[idx]
            
            image_stim.image = image_path
            image_stim.draw()
            win.flip()
            
            # Get response
            trial_clock = core.Clock()
            keys = event.waitKeys(
                maxWait=params['response_window'],
                keyList=RESPONSE_KEYS['number_keys'][:len(params['emotions'])]
            )
            
            # Process response
            if keys:
                rt = trial_clock.getTime()
                response = emotion_keys[keys[0]]
                correct = response == true_emotion
            else:
                rt = None
                response = "no_response"
                correct = False

            if not is_practice:
                # Record data
                data["trial"].append(trial)
                data["image"].append(os.path.basename(image_path))
                data["true_emotion"].append(true_emotion)
                data["response"].append(response)
                data["correct"].append(correct)
                data["rt"].append(rt)

            # Clear screen
            win.flip()

            # Show feedback during practice
            if is_practice:
                if keys:
                    feedback = "Correct!" if correct else f"Incorrect! That was {true_emotion}"
                else:
                    feedback = "Too slow! Please respond faster"
                
                feedback_stim = visual.TextStim(
                    win, 
                    text=feedback, 
                    height=win.size[1]/20
                )
                feedback_stim.draw()
                win.flip()
                core.wait(params['feedback_duration'])

            # ISI
            core.wait(params['isi'])

    # Run practice trials
    practice_text = visual.TextStim(
        win=win, 
        text="Practice Phase", 
        height=win.size[1]/20
    )
    practice_text.draw()
    win.flip()
    core.wait(INSTRUCTION_DURATION)
    
    run_trial_block(params['practice_trials'], is_practice=True)

    # Start main experiment
    exp_text = visual.TextStim(
        win=win,
        text="Main experiment starting...\n\nPress any key to begin",
        height=win.size[1]/20
    )
    exp_text.draw()
    win.flip()
    event.waitKeys()

    # Run main trials
    run_trial_block(params['num_trials'], is_practice=False)

    # Save data
    filename = f"data/emotion_{participant_id}_session{session}.csv"
    df = pd.DataFrame(data)
    df['participant_id'] = participant_id
    df['session'] = session
    df['task'] = 'Emotion'
    save_data(df, filename)

    # Calculate and display summary statistics
    summary = df.groupby("true_emotion").agg({
        "correct": ["mean", "std"],
        "rt": lambda x: np.mean([r for r in x if r is not None])
    }).round(3)

    print("\nTask Summary:")
    print(summary) 