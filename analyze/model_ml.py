import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import glob
import os

# 设置数据文件路径
data_path = r"C:\Users\XX\coffee_behavior\data"

# 假设你有30个被试的文件，命名格式为：
# nback_stats_{id}_session1.csv / nback_stats_{id}_session2.csv
# stroop_{id}_session1_stats_new.csv / stroop_{id}_session2_stats_new.csv
# bart_{id}_session1.csv / bart_{id}_session2.csv
subject_ids = range(11, 30)
data = []

for subj in subject_ids:
    # 使用glob查找每个被试的数据文件
    nback_s1_files = glob.glob(os.path.join(data_path, f'nback_stats_{subj}_session1.csv'))
    nback_s2_files = glob.glob(os.path.join(data_path, f'nback_stats_{subj}_session2.csv'))
    stroop_s1_files = glob.glob(os.path.join(data_path, f'stroop_{subj}_session1_stats_new.csv'))
    stroop_s2_files = glob.glob(os.path.join(data_path, f'stroop_{subj}_session2_stats_new.csv'))
    bart_s1_files = glob.glob(os.path.join(data_path, f'bart_{subj}_session1.csv'))
    bart_s2_files = glob.glob(os.path.join(data_path, f'bart_{subj}_session2.csv'))
    
    # 确保文件存在后再读取
    if nback_s1_files and nback_s2_files and stroop_s1_files and stroop_s2_files and bart_s1_files and bart_s2_files:
        nback_s1 = pd.read_csv(nback_s1_files[0])
        nback_s2 = pd.read_csv(nback_s2_files[0])
        stroop_s1 = pd.read_csv(stroop_s1_files[0])
        stroop_s2 = pd.read_csv(stroop_s2_files[0])
        bart_s1 = pd.read_csv(bart_s1_files[0])
        bart_s2 = pd.read_csv(bart_s2_files[0])
        
        # 计算n-back正确率变化（0-back, 1-back, 2-back）
        nback_feature = {}
        for n in [0, 1, 2]:
            acc_change = nback_s2[nback_s2['n_back'] == n]['accuracy'].mean() - \
                        nback_s1[nback_s1['n_back'] == n]['accuracy'].mean()
            nback_feature[f'nback_{n}_acc_change'] = acc_change
        
        # 计算stroop正确率变化（中文字段）
        stroop_feature = {
            'stroop_neutral_acc_change': stroop_s2['中性词正确率'].mean() - stroop_s1['中性词正确率'].mean(),
            'stroop_congruent_acc_change': stroop_s2['一致条件正确率'].mean() - stroop_s1['一致条件正确率'].mean(),
            'stroop_incongruent_acc_change': stroop_s2['不一致条件正确率'].mean() - stroop_s1['不一致条件正确率'].mean()
        }
        
        # 取第13行作为平均pumps（按你的说明）
        bart_mean_pumps_s1 = bart_s1.loc[12, 'pumps']
        bart_mean_pumps_s2 = bart_s2.loc[12, 'pumps']
        bart_change = bart_mean_pumps_s2 - bart_mean_pumps_s1
        
        # 汇总特征
        features = {**nback_feature, **stroop_feature, 'bart_mean_pumps_change': bart_change}
        features['subject'] = subj
        data.append(features)
    else:
        print(f"Missing files for subject {subj}")

# 转为DataFrame
df = pd.DataFrame(data)

# 设定机器学习标签：是否为冲动增加者（咖啡后pumps增加）
df['label'] = (df['bart_mean_pumps_change'] > 0).astype(int) # 1代表冲动增加，0代表减少或无变化

# 特征与标签
X = df.drop(columns=['subject', 'label', 'bart_mean_pumps_change'])
y = df['label']

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42)
}

results = {}
output_dir = r"C:\Users\XX\coffee_behavior\data\visualizations"
os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

for name, model in models.items():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
    results[name] = {
        'accuracy': np.mean(scores),
        'std': np.std(scores),
        'feature_importance': None
    }
    
    # 先训练模型，然后再计算特征重要性
    model.fit(X_scaled, y)
    
    # 保存特征重要性（仅对树模型有效）
    if hasattr(model, 'feature_importances_'):
        results[name]['feature_importance'] = pd.Series(
            model.feature_importances_, 
            index=X.columns
        ).sort_values(ascending=False)
    else:
        # 非树模型使用置换重要性作为替代
        from sklearn.inspection import permutation_importance
        result = permutation_importance(model, X_scaled, y, n_repeats=10, random_state=42)
        results[name]['feature_importance'] = pd.Series(
            result.importances_mean, 
            index=X.columns
        ).sort_values(ascending=False)

# --------- 图表保存函数 ---------
def save_figure(title, filename):
    plt.savefig(os.path.join(output_dir, filename), 
                dpi=300, 
                bbox_inches='tight', 
                pad_inches=0.2,
                format='png')
    plt.close()

# --------- 可视化部分 ---------
plt.figure(figsize=(15, 10))


# 准确率对比图
plt.subplot(2, 2, 1)
sns.barplot(x=list(results.keys()), y=[v['accuracy'] for v in results.values()])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
save_figure('Model Accuracy Comparison', 'model_accuracy.png')

# 添加标准差置信区间
for i, model in enumerate(results):
    acc = results[model]['accuracy']
    std = results[model]['std']
    plt.errorbar(i, acc, yerr=std, fmt='o', color='black')

# 特征重要性对比图
plt.figure(figsize=(15, 20))
max_features = 10

for idx, (model_name, result) in enumerate(results.items()):
    plt.subplot(len(models), 1, idx+1)
    importance = result['feature_importance'].head(max_features)
    sns.barplot(x=importance.values, y=importance.index)
    plt.title(f'{model_name} Feature Importance (Top {max_features})')
    plt.xlabel('Relative Importance')
    plt.tight_layout()

save_figure('Feature Importance Comparison', 'feature_importance.png')

# --------- 打印分析结果 ---------
print("\nAnalysis Results saved to:", output_dir)
print("Key findings:")
for name, result in results.items():
    print(f"{name}: Accuracy={result['accuracy']:.3f}±{result['std']:.3f}")