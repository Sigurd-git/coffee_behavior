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
        'num_balloons': 30,
        'max_pumps': 128,
        'practice_balloons': 3,
        'reward_per_pump': 0.05,  # $0.05 per pump
        'initial_radius': 50,
        'pump_increment': 2,
        'max_radius': 300,
        'explosion_probabilities': np.linspace(1/128, 1/8, 128),
        'feedback_duration': FEEDBACK_DURATION
    }

    # Save experiment parameters
    exp_info = {
        'participant_id': participant_id,
        'session': session,
        'task': 'BART',
        'parameters': params
    }

    # Create stimuli with standardized size
    balloon = visual.Circle(
        win=win,
        radius=params['initial_radius'],
        fillColor="blue",
        lineColor="white",
        lineWidth=3
    )

    # Create text display with standardized size
    text_display = visual.TextStim(
        win=win,
        height=win.size[1] / 20,  # Smaller text for info display
        color="white",
        pos=(0, -win.size[1]/3)  # Position relative to screen height
    )

    # Show instructions
    instructions = """
    Balloon Pumping Game:
    
    - Press SPACE to pump the balloon
    - Press ENTER to collect your money
    - Be careful! The balloon might pop!
    - Larger balloons = more money, but also more risk
    
    Press any key to start practice.
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
                balloon.radius = params['initial_radius'] + (pumps * params['pump_increment'])
                balloon.draw()
                
                # Show trial info
                text_display.text = (
                    f"Balloon #{trial + 1}\n"
                    f"Current pumps: {pumps}\n"
                    f"Current value: ${pumps * params['reward_per_pump']:.2f}\n"
                    f"Total earned: ${total_earned:.2f}"
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
                    if np.random.random() < params['explosion_probabilities'][pumps-1]:
                        exploded = True
                        break

                    if pumps >= params['max_pumps']:
                        break

            # Trial outcome
            if not exploded:
                earned = pumps * params['reward_per_pump']
                total_earned += earned
                outcome_text = f"You earned ${earned:.2f}!"
            else:
                earned = 0
                outcome_text = "Balloon popped! No money earned."

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
            core.wait(params['feedback_duration'])

        return total_earned

    # Run practice
    practice_text = visual.TextStim(
        win=win, 
        text="Practice Phase", 
        height=win.size[1]/20
    )
    practice_text.draw()
    win.flip()
    core.wait(INSTRUCTION_DURATION)
    
    run_trial_block(params['practice_balloons'], is_practice=True)

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
    total_earned = run_trial_block(params['num_balloons'], is_practice=False)

    # Save data
    filename = f"data/bart_{participant_id}_session{session}.csv"
    df = pd.DataFrame(data)
    df['participant_id'] = participant_id
    df['session'] = session
    df['task'] = 'BART'
    save_data(df, filename)

    # Calculate and display summary statistics
    summary = {
        "Mean Pumps": df["pumps"].mean(),
        "Explosion Rate": df["exploded"].mean() * 100,
        "Total Earned": total_earned,
        "Average Earned Per Balloon": df["earned"].mean(),
    }

    print("\nTask Summary:")
    for key, value in summary.items():
        print(f"{key}: {value:.2f}")