import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from glob import glob

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_nback_correct_rt():
    """分析n-back任务中正确试次的反应时"""
    
    # 读取所有原始数据文件（不是统计文件）
    session1_files = glob('data/nback_[0-9]*_session1.csv')
    session2_files = glob('data/nback_[0-9]*_session2.csv')
    
    # 存储所有被试的数据
    all_data = []
    
    # 处理每个会话的数据
    for files, session in [(session1_files, 1), (session2_files, 2)]:
        for file in files:
            df = pd.read_csv(file)
            
            # 只选择正确的试次和目标试次（target为True且做出了正确反应）
            correct_trials = df[(df['target'] == True) & (df['correct'] == True)]
            
            # 计算每个n-back水平的平均反应时
            for n_back in df['n_back'].unique():
                n_back_trials = correct_trials[correct_trials['n_back'] == n_back]
                if not n_back_trials.empty:
                    # 只选择有效的反应时（不为空且不为0的值）
                    valid_rts = [rt for rt in n_back_trials['rt'] if rt is not None and rt > 0]
                    if valid_rts:
                        mean_rt = np.mean(valid_rts) * 1000  # 转换为毫秒
                        all_data.append({
                            'participant_id': df['participant_id'].iloc[0],
                            'session': session,
                            'n_back': n_back,
                            'mean_rt': mean_rt
                        })
    
    # 转换为DataFrame
    results_df = pd.DataFrame(all_data)
    
    # 计算描述性统计
    desc_stats = results_df.groupby(['session', 'n_back']).agg({
        'mean_rt': ['mean', 'std', 'count']
    }).round(2)
    
    print("\n描述性统计：")
    print(desc_stats)
    
    # 对每个n-back水平进行配对t检验
    print("\n统计检验结果：")
    for n in sorted(results_df['n_back'].unique()):
        # 获取完成了两个session的被试数据
        participants = set(results_df[(results_df['session'] == 1) & (results_df['n_back'] == n)]['participant_id']) & \
                      set(results_df[(results_df['session'] == 2) & (results_df['n_back'] == n)]['participant_id'])
        
        if participants:  # 如果有完成两个session的被试
            # 只选择完成了两个session的被试数据
            session1_rt = []
            session2_rt = []
            for pid in participants:
                s1_rt = results_df[(results_df['session'] == 1) & 
                                 (results_df['n_back'] == n) & 
                                 (results_df['participant_id'] == pid)]['mean_rt'].iloc[0]
                s2_rt = results_df[(results_df['session'] == 2) & 
                                 (results_df['n_back'] == n) & 
                                 (results_df['participant_id'] == pid)]['mean_rt'].iloc[0]
                session1_rt.append(s1_rt)
                session2_rt.append(s2_rt)[[[[[[[[[[[[]]]]]]]]]]]]
            
            if len(session1_rt) > 1:  # 确保有足够的样本进行t检验
                t_stat, p_val = stats.ttest_rel(session1_rt, session2_rt)
                print(f"\n{int(n)}-back 水平:")
                print(f"完成两个session的被试数量: {len(session1_rt)}")
                print(f"t = {t_stat:.3f}, p = {p_val:.3f}")
            else:
                print(f"\n{int(n)}-back 水平: 样本量不足，无法进行统计检验")
    
    # 绘制对比图
    plt.figure(figsize=(10, 6))
    
    # 使用seaborn绘制带误差线的条形图
    sns.barplot(x='n_back', y='mean_rt', hue='session', 
                data=results_df, 
                palette=['lightblue', 'lightgreen'],
                ci=68)
    
    plt.title('n-back任务正确试次反应时对比')
    plt.xlabel('n-back水平')
    plt.ylabel('反应时 (ms)')
    plt.legend(title='实验阶段', labels=['咖啡前', '咖啡后'])
    
    # 添加数值标签
    for i, session in enumerate([1, 2]):
        for j, n in enumerate(sorted(results_df['n_back'].unique())):
            mean_rt = results_df[(results_df['session'] == session) & 
                               (results_df['n_back'] == n)]['mean_rt'].mean()
            x = j + (i - 0.5) * 0.35
            plt.text(x, mean_rt, f'{mean_rt:.0f}', 
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('nback_correct_rt_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    analyze_nback_correct_rt() 