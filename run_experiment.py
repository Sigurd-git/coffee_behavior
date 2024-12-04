import os

# Set QT_QPA_PLATFORM_PLUGIN_PATH before importing psychopy
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = (
    ""  # Force PyQt to find plugins automatically
)
os.environ["QT_DEBUG_PLUGINS"] = "1"  # Enable debug output for Qt plugins

from psychopy import visual, event
from tasks.main import run_experiment


def main():
    # Create window
    win = visual.Window(size=(1024, 1024), fullscr=False, units="pix", color=[0, 0, 0])
    try:
        # Get participant info
        participant_id = input("Enter participant ID: ")
        session = int(input("Enter session (1 = Pre-coffee, 2 = Post-coffee): "))

        # Run experiment
        run_experiment(win, participant_id, session)

    finally:
        # Clean up
        win.close()


if __name__ == "__main__":
    main()
