import numpy as np
from psychopy import visual, core, event
import pandas as pd
from utils.common import show_instructions, save_data
from tasks.config import *
import os
from glob import glob


def run_emotion(win, participant_id, session):
    """
    Emotion Recognition and Rating Task implementation

    Parameters:
    - win: PsychoPy window object
    - participant_id: Participant identifier
    - session: Session number
    """
    # Task parameters
    params = {
        "stim_duration": 6.0,  # Stimulus presentation time in seconds
        "rating_duration": 10.0,  # Maximum time for rating
        "isi": 0.5,  # Inter-stimulus interval
        "images_per_category": 10,
        "stim_size": (800, 600),  # Standard size for all images
        "stim_dirs": {
            "positive": "stimuli/emotions/positive",
            "neutral": "stimuli/emotions/neutral",
            "negative": "stimuli/emotions/negative",
        },
    }
    # # TEST parameters
    # params = {
    #     "stim_duration": 1.0,  # Stimulus presentation time in seconds
    #     "rating_duration": 1.0,  # Maximum time for rating
    #     "isi": 0.5,  # Inter-stimulus interval
    #     "images_per_category": 1,
    #     "stim_size": (800, 600),  # Standard size for all images
    #     "stim_dirs": {
    #         "positive": "stimuli/emotions/positive",
    #         "neutral": "stimuli/emotions/neutral",
    #         "negative": "stimuli/emotions/negative",
    #     },
    # }

    # Initialize data recording
    data = {
        "trial": [],
        "image": [],
        "category": [],
        "phase": [],
        "valence": [],
        "arousal": [],
        "rating_rt": [],
        "participant_id": [],
        "session": [],
    }
    filename = f"data/sub-{participant_id}_ses-{session}_task-emotion.csv"

    def load_images():
        """Load and randomize images from each category"""
        image_paths = []
        categories = []

        for category, directory in params["stim_dirs"].items():
            files = glob(os.path.join(directory, "*.jpg"))
            if not files:
                raise ValueError(f"No jpg files found in {directory}")

            # Randomly select images if more than required
            if len(files) > params["images_per_category"]:
                files = np.random.choice(
                    files, params["images_per_category"], replace=False
                )

            image_paths.extend(files)
            categories.extend([category] * len(files))

        return image_paths, categories

    def show_task_instructions(phase="baseline"):
        """Display task instructions based on the phase"""
        if phase == "baseline":
            instructions = """
            欢迎参加情绪实验。

            在接下来的实验中，您将看到一系列图片。
            请根据您的真实感受对每张图片进行评分：

            效价评分：1(非常不愉快) 到 9(非常愉快)
            唤醒度评分：1(非常平静) 到 9(非常唤醒)

            按空格键继续
            """
        else:  # regulation phase
            instructions = """
            接下来是情绪调节阶段。

            请尝试用不同的角度重新解读图片内容，
            例如，将灾难场景想象为电影场景。

            之后仍需对图片进行效价和唤醒度评分。

            按空格键继续
            """

        instruction_text = visual.TextStim(win, text=instructions, height=30)
        instruction_text.draw()
        win.flip()
        event.waitKeys(keyList=["space"])
        win.flip()

    def get_rating(prompt, instruction):
        """Get rating response from participant using Slider

        Args:
            prompt (str): Rating prompt text
            instruction (str): Instruction text

        Returns:
            tuple: (rating value, response time)
        """
        # Create text stimulus for prompt and instruction
        rating_text = visual.TextStim(
            win, text=f"{prompt}\n\n{instruction}", height=40, pos=(0, 200)
        )

        # Create slider instead of rating scale
        rating_slider = visual.Slider(
            win,
            ticks=(1, 2, 3, 4, 5, 6, 7, 8, 9),  # Show major tick marks
            labels=None,  # No text labels
            granularity=1.0,  # Allow only whole numbers
            size=(500, 50),  # Width and height of slider
            pos=(0, -100),  # Position slightly below center
            style=["rating"],  # Use rating style
        )

        # Get response
        rating_clock = core.Clock()
        while rating_slider.getRating() is None:
            if rating_clock.getTime() > params["rating_duration"]:
                break
            rating_text.draw()
            rating_slider.draw()
            win.flip()

        rating = rating_slider.getRating()
        rt = rating_slider.getRT()

        return (rating or 0, rt or 0)  # Return 0 if no response

    def run_trial_block(image_paths, categories, phase="baseline"):
        """Run a block of trials with ratings"""
        # Randomize trial order
        trial_order = np.random.permutation(len(image_paths))

        for trial, idx in enumerate(trial_order):
            # Present image
            image = visual.ImageStim(
                win, image=image_paths[idx], size=params["stim_size"]
            )
            image.draw()
            win.flip()
            core.wait(params["stim_duration"])

            # Get ratings
            valence, val_rt = get_rating("情绪效价评分", "1 = 非常不愉快, 9 = 非常愉快")
            arousal, aro_rt = get_rating("唤醒度评分", "1 = 非常平静, 9 = 非常唤醒")

            # Record data
            data["trial"].append(trial)
            data["image"].append(os.path.basename(image_paths[idx]))
            data["category"].append(categories[idx])
            data["phase"].append(phase)
            data["valence"].append(valence)
            data["arousal"].append(arousal)
            data["rating_rt"].append(val_rt)
            data["participant_id"].append(participant_id)
            data["session"].append(session)

            # ISI
            win.flip()
            core.wait(params["isi"])

    def run_regulation_phase(negative_images, negative_categories):
        """Run the emotion regulation phase with negative images"""
        show_task_instructions(phase="regulation")
        run_trial_block(negative_images, negative_categories, phase="regulation")

    try:
        # Load all images
        image_paths, categories = load_images()

        # Separate negative images for regulation phase
        negative_indices = [i for i, cat in enumerate(categories) if cat == "negative"]
        negative_images = [image_paths[i] for i in negative_indices]
        negative_categories = [categories[i] for i in negative_indices]

        # Run baseline phase
        show_task_instructions(phase="baseline")
        run_trial_block(image_paths, categories, phase="baseline")

        # Run regulation phase
        run_regulation_phase(negative_images, negative_categories)

        # Save data
        save_data(data, filename)

        return True

    except Exception as e:
        print(f"Error in emotion task: {str(e)}")
        return False
