from psychopy import visual
from tasks.main import run_experiment


def main():
    # Create window
    win = visual.Window(size=(1024, 512), fullscr=False, units="pix", color=[0, 0, 0],pos=(100,100))
    try:
        # Get participant info
        participant_id = input("Enter participant ID: ")
        session = int(input("Enter session (1 = Pre-coffee, 2 = Post-coffee): "))
        # participant_id = "1"
        # session = 1

        # Run experiment
        run_experiment(win, participant_id, session)

    finally:
        # Clean up
        win.close()


if __name__ == "__main__":
    main()
