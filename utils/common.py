"""
Common utilities shared across tasks
"""

import platform

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

def get_system_font():
    """返回基于操作系统的合适字体"""
    if platform.system() == 'Darwin':  # macOS
        # 按优先级尝试不同的字体
        mac_fonts = [
            'Heiti TC',
            'STHeiti',
            'Hiragino Sans GB',
            'Apple LiGothic',
            'Arial Unicode MS'
        ]
        # 返回第一个可用的字体
        return mac_fonts[0]  # 为简单起见直接返回第一个，实际使用时会自动降级到可用字体
    elif platform.system() == 'Windows':
        return 'SimHei'
    else:
        return 'DejaVu Sans' 