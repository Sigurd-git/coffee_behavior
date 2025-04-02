import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Read all nback stats files
nback_stats_paths = sorted(glob.glob('data/nback_stats_*_session*.csv'))
print(f"Found {len(nback_stats_paths)} nback stats files")

# Concatenate all files into a single dataframe
nback_stats_all = pd.concat([pd.read_csv(file) for file in nback_stats_paths], ignore_index=True)


# Filter only task level data (exclude block level)
nback_task_data = nback_stats_all[nback_stats_all['level'] == 'task']

# Create squared n_back term for quadratic model
nback_task_data['n_back_squared'] = nback_task_data['n_back'] ** 2

# Separate data by session
session1_data = nback_task_data[nback_task_data['session'] == 1]  # Pre-coffee
session2_data = nback_task_data[nback_task_data['session'] == 2]  # Post-coffee

# 创建一个新的DataFrame用于分析
df = nback_task_data.copy()

# 添加b项（session2为咖啡）
df['b'] = df['session'].apply(lambda x: 1 if x == 2 else 0)

# y 是 accuracy（转换为0-1范围）
df['y'] = df['accuracy'] / 100

# 根据公式计算 dy/dt
df['dy_dt'] = df['n_back'] + df['b'] * df['y'] - df['y'] ** 3

# 数据标准化处理
scaler = StandardScaler()
df[['n_back', 'y', 'dy_dt']] = scaler.fit_transform(df[['n_back', 'y', 'dy_dt']])

# 准备交叉验证
X = df[['n_back', 'b']].values  # 预测变量
y = df['y'].values  # 目标变量

# 使用K折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 使用scikit-learn的LinearRegression进行交叉验证
model_sk = LinearRegression()
cv_scores = cross_val_score(model_sk, X, y, cv=kf, scoring='r2')

# 输出交叉验证结果
print("Cross-validation R² scores:", cv_scores)
print("Mean CV R²:", np.mean(cv_scores))
print("Standard deviation of CV R²:", np.std(cv_scores))

# 拟合完整数据集以获取系数
model_sk.fit(X, y)
print("Model coefficients:", model_sk.coef_)
print("Model intercept:", model_sk.intercept_)

# 使用statsmodels进行详细分析（用于获取p值等）
model = smf.ols('y ~ n_back + b', data=df).fit()
print(model.summary())

# Create visualization of the results
plt.figure(figsize=(15, 10))

# Plot 1: Scatter plot of dy/dt vs n_back with regression line
plt.subplot(2, 2, 1)
sns.scatterplot(x='n_back', y='dy_dt', hue='session', data=df)
sns.regplot(x='n_back', y='dy_dt', data=df, scatter=False, color='black')
plt.title('dy/dt vs n_back')
plt.xlabel('n_back')
plt.ylabel('dy/dt')

# Plot 2: Scatter plot of dy/dt vs accuracy with regression line
plt.subplot(2, 2, 2)
sns.scatterplot(x='y', y='dy_dt', hue='session', data=df)
plt.title('dy/dt vs Accuracy')
plt.xlabel('Accuracy')
plt.ylabel('dy/dt')

# Plot 3: Coefficient plot
plt.subplot(2, 2, 3)
coefs = model.params
errors = model.bse
coef_names = coefs.index

plt.errorbar(x=range(len(coefs)), y=coefs, yerr=errors, fmt='o', capsize=5)
plt.xticks(range(len(coefs)), coef_names, rotation=45)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Model Coefficients with Error Bars')
plt.ylabel('Standardized Coefficient Value')
plt.grid(alpha=0.3)

# Plot 4: Cross-validation results
plt.subplot(2, 2, 4)
plt.bar(range(len(cv_scores)), cv_scores)
plt.axhline(y=np.mean(cv_scores), color='r', linestyle='-', label=f'Mean R²: {np.mean(cv_scores):.3f}')
plt.xlabel('Fold')
plt.ylabel('R² Score')
plt.title('Cross-Validation R² Scores')
plt.legend()

plt.tight_layout()

# 创建额外的诊断图
plt.figure(figsize=(15, 5))

# Plot 5: Residuals vs Fitted Values
plt.subplot(1, 3, 1)
residuals = model.resid
fitted = model.fittedvalues
plt.scatter(fitted, residuals)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')

# Plot 6: QQ Plot of residuals
plt.subplot(1, 3, 2)
qqplot(residuals, line='s', ax=plt.gca())
plt.title('Q-Q Plot of Residuals')

# Plot 7: Actual vs Predicted
plt.subplot(1, 3, 3)
plt.scatter(df['dy_dt'], fitted)
min_val = min(df['dy_dt'].min(), fitted.min())
max_val = max(df['dy_dt'].max(), fitted.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual dy/dt')
plt.ylabel('Predicted dy/dt')

plt.tight_layout()

# 相位图
plt.figure(figsize=(10, 6))
# 创建y值网格
y_vals = np.linspace(0, 1, 100)

# 计算不同参数组合的dy/dt
for n in [0, 1, 2]:  # 使用实际的n_back值
    for b in [0, 1]:
        # 使用原始尺度（非标准化）计算dy/dt
        dy_dt = n + b * y_vals - y_vals**3
        plt.plot(y_vals, dy_dt, label=f'n={n}, b={b}')

# 添加平衡线
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title('Phase Portrait: dy/dt vs y')
plt.xlabel('y (Accuracy)')
plt.ylabel('dy/dt')
plt.legend()
plt.grid(True, alpha=0.3)

# 在相位图上添加实际数据点
# 使用原始数据（非标准化）进行绘制
original_y = nback_task_data['accuracy'] / 100  # 假设原始准确率是百分比
original_dy_dt = nback_task_data['n_back'] + (nback_task_data['session'] == 2) * original_y - original_y**3
plt.scatter(original_y, original_dy_dt, c=nback_task_data['session'], cmap='coolwarm', alpha=0.6)

plt.tight_layout()
plt.show()
