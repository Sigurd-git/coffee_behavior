import pandas as pd
import numpy as np

# 读取原始数据
participant_id = 3  # 替换为实际的参与者ID
session = 1  # 替换为实际的会话编号
raw_filename = f"data/nback_{participant_id}_session{session}.csv"
df = pd.read_csv(raw_filename)

# 计算新的正确率
def calculate_new_accuracy(df):
    """计算目标刺激的正确率"""
    stats = []

    # 计算每个block的统计数据
    block_stats = df.groupby(['n_back', 'block']).agg({
        'correct': 'mean',  # 计算平均正确率
        'rt': lambda x: np.mean([r for r in x if r is not None])  # 平均反应时（排除None）
    }).reset_index()

    # 添加block级别的统计
    for _, row in block_stats.iterrows():
        # 计算目标刺激的正确率
        target_data = df[(df['n_back'] == row['n_back']) & (df['block'] == row['block']) & (df['target'] == True)]
        if not target_data.empty:
            target_correct_count = target_data['correct'].sum()  # 统计正确的目标刺激数量
            target_count = target_data.shape[0]  # 统计目标刺激的总数量
            accuracy = (target_correct_count / target_count) * 100  # 计算目标刺激的正确率
        else:
            accuracy = 0  # 如果没有目标刺激，正确率为0

        stats.append({
            'level': 'block',
            'n_back': row['n_back'],
            'block': row['block'],
            'accuracy': accuracy,  # 使用新的正确率
            'mean_rt': row['rt'],
            'participant_id': df['participant_id'].iloc[0],
            'session': df['session'].iloc[0]
        })

    # 计算每个n-back水平的总体统计
    task_stats = df.groupby('n_back').agg({
        'correct': 'mean',
        'rt': lambda x: np.mean([r for r in x if r is not None])
    }).reset_index()

    # 添加task级别的统计
    for _, row in task_stats.iterrows():
        # 计算目标刺激的正确率
        target_data = df[(df['n_back'] == row['n_back']) & (df['target'] == True)]
        if not target_data.empty:
            target_correct_count = target_data['correct'].sum()  # 统计正确的目标刺激数量
            target_count = target_data.shape[0]  # 统计目标刺激的总数量
            accuracy = (target_correct_count / target_count) * 100  # 计算目标刺激的正确率
        else:
            accuracy = 0  # 如果没有目标刺激，正确率为0

        stats.append({
            'level': 'task',
            'n_back': row['n_back'],
            'block': 'all',  # 表示所有block的平均
            'accuracy': accuracy,  # 使用新的正确率
            'mean_rt': row['rt'],
            'participant_id': df['participant_id'].iloc[0],
            'session': df['session'].iloc[0]
        })

    return pd.DataFrame(stats)

# 计算新的统计数据
new_stats_df = calculate_new_accuracy(df)

# 保存新的统计数据
new_stats_filename = f"data/nback_new_stats_{participant_id}_session{session}.csv"
new_stats_df.to_csv(new_stats_filename, index=False)

print(f"New statistics saved to {new_stats_filename}") 