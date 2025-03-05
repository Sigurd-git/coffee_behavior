import numpy as np
from psychopy import visual, core, event, monitors
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
    # 设置显示器参数
    mon = monitors.Monitor('testMonitor')
    mon.setSizePix([1440, 900])  # 设置实际的屏幕分辨率
    mon.setWidth(30)  # 设置显示器的物理宽度（厘米）
    mon.setDistance(57)  # 设置观看距离（厘米）
    win.monitor = mon

    # Task parameters
    params = {
        "stim_duration": 4.0,  # Stimulus presentation time in seconds
        "regulation_duration": 6.0,  # Duration for regulation phase
        "rating_duration": 10.0,  # Maximum time for rating
        "isi": 0.5,  # Inter-stimulus interval
        "images_per_category": 10,
        "stim_size": (1000, 750),  # Standard size for all images
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
        """Load and split images into pre and post coffee sessions"""
        pre_coffee_images = []
        pre_coffee_categories = []
        post_coffee_images = []
        post_coffee_categories = []

        # 加载并分组图片
        for category, directory in params["stim_dirs"].items():
            # 获取所有图片并排序，确保顺序一致
            files = sorted(glob(os.path.join(directory, "*.jpg")))
            if not files:
                raise ValueError(f"No jpg files found in {directory}")
            
            # 检查图片数量
            total_images = len(files)
            if total_images != 40:
                raise ValueError(f"Category {category} must have exactly 40 images, but found {total_images}")
            
            # 严格划分为前20和后20张
            pre_coffee = files[:20]  # 前20张用于咖啡前
            post_coffee = files[20:40]  # 后20张用于咖啡后
            
            # 验证分组数量
            if len(pre_coffee) != 20 or len(post_coffee) != 20:
                raise ValueError(f"Failed to split {category} images into equal groups of 20")
            
            # 添加到相应的列表中
            pre_coffee_images.extend(pre_coffee)
            pre_coffee_categories.extend([category] * 20)
            post_coffee_images.extend(post_coffee)
            post_coffee_categories.extend([category] * 20)

        # 最终验证
        if len(pre_coffee_images) != 60 or len(post_coffee_images) != 60:
            raise ValueError("Each session must have exactly 60 images (20 per category)")

        # 打印验证信息
        print("\nImage distribution:")
        print("Pre-coffee session:")
        for cat in params["stim_dirs"].keys():
            print(f"  {cat}: {pre_coffee_categories.count(cat)} images")
        print("\nPost-coffee session:")
        for cat in params["stim_dirs"].keys():
            print(f"  {cat}: {post_coffee_categories.count(cat)} images")

        return pre_coffee_images, pre_coffee_categories, post_coffee_images, post_coffee_categories

    def show_task_instructions(phase="baseline"):
        """Display task instructions based on the phase"""
        if phase == "baseline":
            instructions = """
欢迎参加情绪实验，开始前您可作适当休息。

在接下来的实验中，您将看到一系列图片。
图片将呈现6秒，之后10秒时间您需要
对图片进行效价：情绪的正负性体验
和唤醒度：情绪的激活程度的评分。

请根据您的真实感受对每张图片进行评分：

效价评分: 1(让您感觉非常不愉快) 到 9(让您感觉非常愉快)
唤醒度评分: 1(让您感到非常平静) 到 9(让您感到非常兴奋、紧张或激动)

评分好后，可切换至下一张。当您理解并准备好后，请按空格键继续。"""
        else:  # regulation phase
            instructions = """
接下来是情绪调节阶段。

请尝试用不同的角度重新解读负性图片内容，
例如，将灾难场景想象为电影场景，
都是虚构制作或由演员演绎。

之后仍需对图片进行效价和唤醒度评分。

按空格键继续"""

        instruction_text = visual.TextStim(
            win, 
            text=instructions, 
            height=30,
            wrapWidth=1000,  # 增加文本换行宽度以确保更好的排版
            alignText='center',  # 文本居中对齐
            anchorHoriz='center',  # 水平锚点居中
            anchorVert='center',  # 垂直锚点居中
            font='SimHei',
            color='white'
        )
        instruction_text.draw()
        win.flip()
        event.waitKeys(keyList=["space"])
        win.flip()

    def get_ratings(image):
        """同时获取效价和唤醒度评分，并同时呈现图片"""
        # 保持图片在原始位置，不设置pos
        
        # 创建半透明黑色背景
        rating_background = visual.Rect(
            win,
            width=850,  # Slightly wider than the rating slider (800)
            height=300,
            pos=(0, -200),  # Moved up by 100 pixels (10% of typical 1000px height)
            fillColor='black',
            opacity=0.5
        )

        # 创建"next"按钮
        next_button = visual.ButtonStim(
            win,
            text="next",
            pos=(0, -350),  # Moved up by 100 pixels
            size=(100, 40),
            fillColor="darkgrey",
            borderColor="black",
            borderWidth=2,
            color="white",
            letterHeight=20
        )

        # 创建效价评分相关元素
        valence_text = visual.TextStim(
            win, 
            text=u"情绪效价评分\n1 = 非常不愉快, 9 = 非常愉快", 
            height=30, 
            pos=(0, -100),  # Moved up by 100 pixels
            font='SimHei',
            color='white'
        )
        
        # 修改滑块设置，确保可见
        valence_slider = visual.Slider(
            win,
            ticks=(1, 2, 3, 4, 5, 6, 7, 8, 9),
            granularity=1.0,
            size=(800, 30),
            pos=(0, -150),  # Moved up by 100 pixels
            style=["rating"],
            borderColor='white',  # 设置边框颜色
            fillColor='white',  # 设置填充颜色
            color='white',  # 设置标记颜色
            markerColor='red',  # 设置滑块标记颜色
            lineColor='white',  # 设置线条颜色
        )
        
        valence_left = visual.TextStim(
            win,
            text="1",
            pos=(-430, -150),  # Moved up by 100 pixels
            color='white',
            height=20
        )
        valence_right = visual.TextStim(
            win,
            text="9",
            pos=(430, -150),  # Moved up by 100 pixels
            color='white',
            height=20
        )

        # 创建唤醒度评分相关元素
        arousal_text = visual.TextStim(
            win, 
            text=u"唤醒度评分\n1 = 非常平静, 9 = 非常唤醒", 
            height=30, 
            pos=(0, -200),  # Moved up by 100 pixels
            font='SimHei',
            color='white'
        )
        
        # 修改滑块设置，确保可见
        arousal_slider = visual.Slider(
            win,
            ticks=(1, 2, 3, 4, 5, 6, 7, 8, 9),
            granularity=1.0,
            size=(800, 30),
            pos=(0, -250),  # Moved up by 100 pixels
            style=["rating"],
            borderColor='white',  # 设置边框颜色
            fillColor='white',  # 设置填充颜色
            color='white',  # 设置标记颜色
            markerColor='red',  # 设置滑块标记颜色
            lineColor='white',  # 设置线条颜色
        )
        
        arousal_left = visual.TextStim(
            win,
            text="1",
            pos=(-430, -250),  # Moved up by 100 pixels
            color='white',
            height=20
        )
        arousal_right = visual.TextStim(
            win,
            text="9",
            pos=(430, -250),  # Moved up by 100 pixels
            color='white',
            height=20
        )

        rating_clock = core.Clock()
        while rating_clock.getTime() < params["rating_duration"]:
            # 保持图片在原始位置
            image.draw()
            rating_background.draw()
            valence_text.draw()
            valence_slider.draw()
            valence_left.draw()
            valence_right.draw()
            arousal_text.draw()
            arousal_slider.draw()
            arousal_left.draw()
            arousal_right.draw()
            next_button.draw()
            win.flip()

            if next_button.isClicked:
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
            text="请将图片想象成虚构制作或由演员演绎",
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

    def run_regulation_phase(images, categories):
        """运行情绪调节阶段"""
        show_task_instructions(phase="regulation")
        
        # 使用enumerate来追踪trial编号
        for trial, (image_path, category) in enumerate(zip(images, categories)):
            # 创建图片刺激
            image = visual.ImageStim(
                win,
                image_path,
                size=params["stim_size"]
            )
            
            regulation_text = visual.TextStim(
                win,
                text="请将图片想象成电影场景",
                pos=(0, 300),
                height=30,
                color='white',
                font='SimHei'
            )
            
            # 呈现图片和提示语规定的时间
            for frame in range(int(params["regulation_duration"] * 60)):  # 假设60Hz刷新率
                image.draw()
                regulation_text.draw()
                win.flip()
            
            # 获取评分
            valence, arousal, rt = get_ratings(image)
            
            # 使用与其他函数相同的数据记录方式
            data["trial"].append(trial)
            data["image"].append(os.path.basename(image_path))
            data["category"].append(category)
            data["phase"].append("regulation")
            data["valence"].append(valence)
            data["arousal"].append(arousal)
            data["rating_rt"].append(rt)
            data["participant_id"].append(participant_id)
            data["session"].append(session)

            # ISI
            win.flip()
            core.wait(params["isi"])

    # 在保存数据之前添加统计分析
    def calculate_emotion_stats(df):
        """计算每个条件下的平均效价和唤醒度"""
        stats = []
        
        # 遍历所有可能的组合
        for phase in ['baseline', 'regulation']:
            for category in ['positive', 'neutral', 'negative']:
                # 获取该条件下的数据
                condition_data = df[(df['phase'] == phase) & (df['category'] == category)]
                
                if not condition_data.empty:  # 只处理有数据的条件
                    mean_valence = condition_data['valence'].mean()
                    mean_arousal = condition_data['arousal'].mean()
                    
                    stats.append({
                        'phase': phase,
                        'category': category,
                        'mean_valence': mean_valence,
                        'mean_arousal': mean_arousal,
                        'participant_id': df['participant_id'].iloc[0],
                        'session': df['session'].iloc[0]
                    })
        
        return pd.DataFrame(stats)

    try:
        # Load all images and get pre/post coffee sets
        pre_coffee_images, pre_coffee_categories, post_coffee_images, post_coffee_categories = load_images()

        # 根据session选择使用哪组图片
        if session == 1:  # 咖啡前
            # 创建随机顺序，但保持图片和类别对应关系
            trial_order = np.random.permutation(len(pre_coffee_images))
            current_images = [pre_coffee_images[i] for i in trial_order]
            current_categories = [pre_coffee_categories[i] for i in trial_order]
        else:  # 咖啡后
            # 创建随机顺序，但保持图片和类别对应关系
            trial_order = np.random.permutation(len(post_coffee_images))
            current_images = [post_coffee_images[i] for i in trial_order]
            current_categories = [post_coffee_categories[i] for i in trial_order]

        print(f"Session {session}: Using {len(current_images)} images")
        print("Categories distribution:", 
              {cat: current_categories.count(cat) for cat in set(current_categories)})

        # 运行基线阶段
        show_task_instructions(phase="baseline")
        run_trial_block(current_images, current_categories, phase="baseline")

        # 找出当前session中的负性图片用于调节阶段
        negative_indices = [i for i, cat in enumerate(current_categories) if cat == "negative"]
        negative_images = [current_images[i] for i in negative_indices]
        negative_categories = [current_categories[i] for i in negative_indices]

        # 运行调节阶段
        run_regulation_phase(negative_images, negative_categories)

        # 将原始数据转换为DataFrame
        df = pd.DataFrame(data)

        # 计算统计数据
        stats_df = calculate_emotion_stats(df)

        # 保存原始数据
        raw_filename = f"data/emotion_{participant_id}_session{session}.csv"
        save_data(df, raw_filename)

        # 保存统计数据
        stats_filename = f"data/emotion_stats_{participant_id}_session{session}.csv"
        save_data(stats_df, stats_filename)

        # 打印统计结果
        print("\n情绪评分统计:")
        for _, row in stats_df.iterrows():
            print(f"\n{row['phase']} - {row['category']}:")
            print(f"  平均效价: {row['mean_valence']:.2f}")
            print(f"  平均唤醒度: {row['mean_arousal']:.2f}")

        return True

    except Exception as e:
        print(f"Error in emotion task: {str(e)}")
        return False
