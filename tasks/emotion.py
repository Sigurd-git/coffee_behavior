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
        "stim_duration": 4.0,  # Stimulus presentation time in seconds
        "regulation_duration": 6.0,  # Duration for regulation phase
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

            效价评分: 1(非常不愉快) 到 9(非常愉快)
            唤醒度评分: 1(非常平静) 到 9(非常唤醒)
            控制感评分: 1(受控制) 到 9(有掌控感)
            
            按空格键继续
            """
        else:  # regulation phase
            instructions = """
            接下来是情绪调节阶段。

            请尝试用不同的角度重新解读负性图片内容，
            例如，将灾难场景想象为电影场景，都是虚构制作或由演员演绎。

            之后仍需对图片进行效价和唤醒度评分。

            按空格键继续
            """

        instruction_text = visual.TextStim(win, text=instructions, height=30)
        instruction_text.draw()
        win.flip()
        event.waitKeys(keyList=["space"])
        win.flip()

    def get_ratings(image):
        """同时获取效价和唤醒度评分，并同时呈现图片"""
        # 创建"下一张"按钮
        next_button = visual.ButtonStim(
            win,
            text="下一张",
            pos=(0, -250),
            size=(100, 40),
            buttonColors=["darkgrey", "grey"]
        )

        # 创建效价和唤醒度滑动条
        valence_text = visual.TextStim(
            win, text="情绪效价评分\n1 = 非常不愉快, 9 = 非常愉快", 
            height=30, pos=(0, 100)
        )
        valence_slider = visual.Slider(
            win,
            ticks=(1, 2, 3, 4, 5, 6, 7, 8, 9),
            granularity=1.0,
            size=(500, 50),
            pos=(0, 50),
            style=["rating"]
        )

        arousal_text = visual.TextStim(
            win, text="唤醒度评分\n1 = 非常平静, 9 = 非常唤醒", 
            height=30, pos=(0, -50)
        )
        arousal_slider = visual.Slider(
            win,
            ticks=(1, 2, 3, 4, 5, 6, 7, 8, 9),
            granularity=1.0,
            size=(500, 50),
            pos=(0, -100),
            style=["rating"]
        )

        rating_clock = core.Clock()
        while rating_clock.getTime() < params["rating_duration"]:
            # 同时呈现图片和评分界面
            image.draw()
            valence_text.draw()
            valence_slider.draw()
            arousal_text.draw()
            arousal_slider.draw()
            next_button.draw()
            win.flip()

            # 检查是否点击了"下一张"按钮
            if next_button.buttonPressed:
                if valence_slider.getRating() is not None and arousal_slider.getRating() is not None:
                    break

        return (
            valence_slider.getRating() or 0,
            arousal_slider.getRating() or 0,
            rating_clock.getTime()
        )

    def run_trial_block(image_paths, categories, phase="baseline"):
        """运行试次"""
        trial_order = np.random.permutation(len(image_paths))

        for trial, idx in enumerate(trial_order):
            # 创建并呈现图片刺激4秒
            image = visual.ImageStim(
                win, image=image_paths[idx], size=params["stim_size"]
            )
            image.draw()
            win.flip()
            core.wait(params["stim_duration"])

            # 获取评分（同时呈现图片）
            valence, arousal, rt = get_ratings(image)

            # 记录数据
            data["trial"].append(trial)
            data["image"].append(os.path.basename(image_paths[idx]))
            data["category"].append(categories[idx])
            data["phase"].append(phase)
            data["valence"].append(valence)
            data["arousal"].append(arousal)
            data["rating_rt"].append(rt)
            data["participant_id"].append(participant_id)
            data["session"].append(session)

            # ISI
            win.flip()
            core.wait(params["isi"])

    def run_regulation_trial(image_paths, categories):
        """运行情绪调节阶段的试次"""
        trial_order = np.random.permutation(len(image_paths))
        
        # 创建调节指导语
        regulation_instruction = visual.TextStim(
            win,
            text="请将灾难想象成电影场景",
            height=30,
            pos=(0, 250)  # 将文字放在图片上方
        )

        for trial, idx in enumerate(trial_order):
            # 创建图片刺激
            image = visual.ImageStim(
                win, image=image_paths[idx], size=params["stim_size"]
            )
            
            # 同时呈现图片和调节指导语6秒
            for frame in range(int(params["regulation_duration"] * 60)):  # 假设60Hz刷新率
                image.draw()
                regulation_instruction.draw()
                win.flip()

            # 获取评分（同时呈现图片）
            valence, arousal, rt = get_ratings(image)

            # 记录数据
            data["trial"].append(trial)
            data["image"].append(os.path.basename(image_paths[idx]))
            data["category"].append(categories[idx])
            data["phase"].append("regulation")
            data["valence"].append(valence)
            data["arousal"].append(arousal)
            data["rating_rt"].append(rt)
            data["participant_id"].append(participant_id)
            data["session"].append(session)

            # ISI
            win.flip()
            core.wait(params["isi"])

    def run_regulation_phase(negative_images, negative_categories):
        """运行情绪调节阶段"""
        show_task_instructions(phase="regulation")
        run_regulation_trial(negative_images, negative_categories)

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

        # Run regulation phase (only with negative images)
        run_regulation_phase(negative_images, negative_categories)

        # Save data
        filename = f"data/emotion_{participant_id}_session{session}.csv"
        save_data(data, filename)

        return True

    except Exception as e:
        print(f"Error in emotion task: {str(e)}")
        return False
