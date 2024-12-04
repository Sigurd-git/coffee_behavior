"""
Common utilities shared across tasks
"""

def show_instructions(win, text):
    """Display instructions and wait for key press"""
    from psychopy import visual, event
    
    instructions = visual.TextStim(
        win=win,
        text=text,
        height=30,
        wrapWidth=800
    )
    instructions.draw()
    win.flip()
    event.waitKeys()

def save_data(data, filename):
    """Save task data to CSV file"""
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False) 