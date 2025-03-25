import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.gofplots import qqplot

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

# Build quadratic models for each session
session1_model = smf.ols('accuracy ~ n_back + n_back_squared', data=session1_data).fit()
session2_model = smf.ols('accuracy ~ n_back + n_back_squared', data=session2_data).fit()

# Print model summaries
print("\n咖啡前(Session 1)模型结果:")
print(session1_model.summary())

print("\n咖啡后(Session 2)模型结果:")
print(session2_model.summary())

# Extract coefficient significance
s1_coef = session1_model.params
s1_pvalues = session1_model.pvalues
s1_conf_int = session1_model.conf_int()

s2_coef = session2_model.params
s2_pvalues = session2_model.pvalues
s2_conf_int = session2_model.conf_int()

# Create a detailed coefficient table
coef_table = pd.DataFrame({
    'Pre-coffee Coef': s1_coef,
    'Pre-coffee p-value': s1_pvalues,
    'Pre-coffee 95% CI Lower': s1_conf_int[0],
    'Pre-coffee 95% CI Upper': s1_conf_int[1],
    'Post-coffee Coef': s2_coef,
    'Post-coffee p-value': s2_pvalues,
    'Post-coffee 95% CI Lower': s2_conf_int[0],
    'Post-coffee 95% CI Upper': s2_conf_int[1],
    'Difference': s2_coef - s1_coef
})

# Save coefficient table to CSV
coef_table.round(4).to_csv('nback_model_coefficients.csv')
print("系数详细分析已保存到 nback_model_coefficients.csv")

# Add significance indicators
def add_significance_indicator(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"

# Create significance table
significance_data = []
for param in s1_coef.index:
    s1_sig = add_significance_indicator(s1_pvalues[param])
    s2_sig = add_significance_indicator(s2_pvalues[param])
    significance_data.append({
        'Parameter': param,
        'Pre-coffee Coef': f"{s1_coef[param]:.4f} {s1_sig}",
        'Post-coffee Coef': f"{s2_coef[param]:.4f} {s2_sig}",
        'Difference': f"{(s2_coef[param] - s1_coef[param]):.4f}"
    })

significance_table = pd.DataFrame(significance_data)
significance_table.to_csv('nback_model_significance.csv', index=False)
print("系数显著性标记已保存到 nback_model_significance.csv")

# Calculate average accuracy by n_back level for each session
avg_accuracy = nback_task_data.groupby(['session', 'n_back'])['accuracy'].mean().reset_index()
avg_accuracy_wide = avg_accuracy.pivot(index='n_back', columns='session', values='accuracy')
avg_accuracy_wide.columns = ['Pre-coffee', 'Post-coffee']
avg_accuracy_wide['Difference'] = avg_accuracy_wide['Post-coffee'] - avg_accuracy_wide['Pre-coffee']

# Save average accuracy to CSV
avg_accuracy_wide.to_csv('nback_average_accuracy.csv')
print("各难度级别平均准确率对比已保存到 nback_average_accuracy.csv")

# Extract model summary statistics
model_stats = pd.DataFrame({
    'Metric': ['R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)', 'AIC', 'BIC'],
    'Pre-coffee': [
        session1_model.rsquared,
        session1_model.rsquared_adj,
        session1_model.fvalue,
        session1_model.f_pvalue,
        session1_model.aic,
        session1_model.bic
    ],
    'Post-coffee': [
        session2_model.rsquared,
        session2_model.rsquared_adj,
        session2_model.fvalue,
        session2_model.f_pvalue,
        session2_model.aic,
        session2_model.bic
    ]
})

# Save model statistics to CSV
model_stats.to_csv('nback_model_statistics.csv', index=False)
print("模型统计信息已保存到 nback_model_statistics.csv")

# Visualize the data and model predictions
plt.figure(figsize=(12, 6))

# Create a range of n_back values for prediction
x_range = np.linspace(0, 2, 100)
x_range_squared = x_range ** 2

# Calculate predicted values
s1_predicted = session1_model.params[0] + session1_model.params[1] * x_range + session1_model.params[2] * x_range_squared
s2_predicted = session2_model.params[0] + session2_model.params[1] * x_range + session2_model.params[2] * x_range_squared

# Plot session 1 data and model
plt.subplot(1, 2, 1)
sns.scatterplot(x='n_back', y='accuracy', data=session1_data, color='blue', alpha=0.7, s=80)
plt.plot(x_range, s1_predicted, 'b-', label='Quadratic Model')
plt.title('Pre-coffee (Session 1)', fontsize=14)
plt.xlabel('N-back Level', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(60, 100)
plt.grid(True, alpha=0.3)

# Plot session 2 data and model
plt.subplot(1, 2, 2)
sns.scatterplot(x='n_back', y='accuracy', data=session2_data, color='green', alpha=0.7, s=80)
plt.plot(x_range, s2_predicted, 'g-', label='Quadratic Model')
plt.title('Post-coffee (Session 2)', fontsize=14)
plt.xlabel('N-back Level', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(60, 100)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nback_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Compare model parameters
params_comparison = pd.DataFrame({
    'Pre-coffee': session1_model.params,
    'Post-coffee': session2_model.params,
    'Difference': session2_model.params - session1_model.params
})

print("\n模型参数对比:")
print(params_comparison)