"""
Taylor图表分析脚本
Taylor Diagram Analysis Script

用于创建Taylor图表比较模型性能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import matplotlib.patches as patches

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

def taylor_statistics(reference, model):
    """计算Taylor图表所需的统计量"""
    # 标准差
    ref_std = np.std(reference)
    mod_std = np.std(model)
    
    # 相关系数
    correlation = np.corrcoef(reference, model)[0, 1]
    
    # 中心化均方根差
    ref_mean = np.mean(reference)
    mod_mean = np.mean(model)
    centered_rms = np.sqrt(np.mean((model - mod_mean - (reference - ref_mean))**2))
    
    return ref_std, mod_std, correlation, centered_rms

def create_taylor_diagram():
    """创建Taylor图表"""
    
    # 1. 加载数据和训练模型
    print("加载数据和训练模型...")
    data = pd.read_csv('dateset.csv')
    
    # 移除最后一列（如果为空）
    if data.columns[-1] == '':
        data = data.drop(data.columns[-1], axis=1)
    
    # 处理缺失值
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if data[col].isnull().any():
            median_val = data[col].median()
            data[col].fillna(median_val, inplace=True)
    
    # 分离特征和目标变量
    feature_columns = [col for col in data.columns if col != 'VS']
    X = data[feature_columns]
    y = data['VS']
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    models = {
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42)
    }
    
    predictions = {}
    
    for name, model in models.items():
        print(f"训练 {name}...")
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        predictions[name] = {
            'train': train_pred,
            'test': test_pred
        }
    
    # 2. 计算Taylor统计量
    print("计算Taylor统计量...")
    
    train_stats = []
    test_stats = []
    
    for model_name, preds in predictions.items():
        # 训练集统计
        ref_std_train, mod_std_train, corr_train, rms_train = taylor_statistics(
            y_train, preds['train']
        )
        train_stats.append({
            'name': model_name,
            'std_ratio': mod_std_train / ref_std_train,
            'correlation': corr_train,
            'rms': rms_train / ref_std_train
        })
        
        # 测试集统计
        ref_std_test, mod_std_test, corr_test, rms_test = taylor_statistics(
            y_test, preds['test']
        )
        test_stats.append({
            'name': model_name,
            'std_ratio': mod_std_test / ref_std_test,
            'correlation': corr_test,
            'rms': rms_test / ref_std_test
        })
    
    # 3. 创建Taylor图表
    print("创建Taylor图表...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(projection='polar'))
      # 绘制训练集Taylor图
    ax1.set_title('Training Set Taylor Diagram', pad=20, fontsize=14, fontfamily='Times New Roman')
    
    # 设置网格和标签
    ax1.set_thetamax(90)
    ax1.set_rlabel_position(0)
    ax1.grid(True)
    
    # 标准差比值的同心圆
    std_ratios = [0.5, 1.0, 1.5, 2.0]
    for ratio in std_ratios:
        circle = plt.Circle((0, 0), ratio, fill=False, color='lightgray', linestyle='--', alpha=0.5)
        ax1.add_patch(circle)
      # 相关系数线
    corr_levels = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    for corr in corr_levels:
        theta = np.arccos(corr)
        ax1.plot([theta, theta], [0, 2.5], 'gray', alpha=0.5, linestyle='-', linewidth=0.5)
        ax1.text(theta, 2.6, f'{corr:.2f}', ha='center', va='bottom', fontsize=8, fontfamily='Times New Roman')
      # 绘制参考点（观测值）
    ax1.plot(0, 1, 'ro', markersize=10, label='Reference')
    
    # 绘制模型点
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    for i, stat in enumerate(train_stats):
        theta = np.arccos(stat['correlation'])
        r = stat['std_ratio']
        ax1.plot(theta, r, 'o', color=colors[i], markersize=8, label=stat['name'])
        ax1.text(theta, r + 0.1, stat['name'], ha='center', va='bottom', fontsize=9, fontfamily='Times New Roman')
    
    ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), prop={'family': 'Times New Roman'})
    ax1.set_ylim(0, 2.5)
      # 绘制测试集Taylor图
    ax2.set_title('Test Set Taylor Diagram', pad=20, fontsize=14, fontfamily='Times New Roman')
    
    # 设置网格和标签
    ax2.set_thetamax(90)
    ax2.set_rlabel_position(0)
    ax2.grid(True)
    
    # 标准差比值的同心圆
    for ratio in std_ratios:
        circle = plt.Circle((0, 0), ratio, fill=False, color='lightgray', linestyle='--', alpha=0.5)
        ax2.add_patch(circle)
      # 相关系数线
    for corr in corr_levels:
        theta = np.arccos(corr)
        ax2.plot([theta, theta], [0, 2.5], 'gray', alpha=0.5, linestyle='-', linewidth=0.5)
        ax2.text(theta, 2.6, f'{corr:.2f}', ha='center', va='bottom', fontsize=8, fontfamily='Times New Roman')
    
    # 绘制参考点（观测值）
    ax2.plot(0, 1, 'ro', markersize=10, label='Reference')
    
    # 绘制模型点
    for i, stat in enumerate(test_stats):
        theta = np.arccos(stat['correlation'])
        r = stat['std_ratio']
        ax2.plot(theta, r, 'o', color=colors[i], markersize=8, label=stat['name'])
        ax2.text(theta, r + 0.1, stat['name'], ha='center', va='bottom', fontsize=9, fontfamily='Times New Roman')
    
    ax2.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), prop={'family': 'Times New Roman'})
    ax2.set_ylim(0, 2.5)
    
    plt.tight_layout()
    plt.savefig('taylor_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. 创建标准的Taylor图（直角坐标系版本）
    print("创建直角坐标系Taylor图...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
      # 训练集
    ax1.set_title('Training Set Taylor Statistics', fontsize=14, fontfamily='Times New Roman')
      # 绘制标准差比值 vs 相关系数
    train_corrs = [stat['correlation'] for stat in train_stats]
    train_std_ratios = [stat['std_ratio'] for stat in train_stats]
    train_names = [stat['name'] for stat in train_stats]
    
    ax1.scatter(train_corrs, train_std_ratios, s=100, alpha=0.7)
    for i, name in enumerate(train_names):
        ax1.annotate(name, (train_corrs[i], train_std_ratios[i]), 
                    xytext=(5, 5), textcoords='offset points', fontfamily='Times New Roman')
    
    # 添加参考线
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Observed Std')
    ax1.axvline(x=1, color='red', linestyle='--', alpha=0.7)
    
    ax1.set_xlabel('Correlation', fontfamily='Times New Roman')
    ax1.set_ylabel('Standard Deviation Ratio', fontfamily='Times New Roman')
    ax1.grid(True, alpha=0.3)
    ax1.legend(prop={'family': 'Times New Roman'})
      # 测试集
    ax2.set_title('Test Set Taylor Statistics', fontsize=14, fontfamily='Times New Roman')
    
    test_corrs = [stat['correlation'] for stat in test_stats]
    test_std_ratios = [stat['std_ratio'] for stat in test_stats]
    test_names = [stat['name'] for stat in test_stats]
    
    ax2.scatter(test_corrs, test_std_ratios, s=100, alpha=0.7)
    for i, name in enumerate(test_names):
        ax2.annotate(name, (test_corrs[i], test_std_ratios[i]), 
                    xytext=(5, 5), textcoords='offset points', fontfamily='Times New Roman')
    
    # 添加参考线
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Observed Std')
    ax2.axvline(x=1, color='red', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Correlation', fontfamily='Times New Roman')
    ax2.set_ylabel('Standard Deviation Ratio', fontfamily='Times New Roman')
    ax2.grid(True, alpha=0.3)
    ax2.legend(prop={'family': 'Times New Roman'})
    
    plt.tight_layout()
    plt.savefig('taylor_statistics_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. 保存统计结果
    print("保存Taylor统计结果...")
    
    taylor_results = []
    
    for i, stat in enumerate(train_stats):
        taylor_results.append({
            'Model': stat['name'],
            'Dataset': 'Train',
            'Correlation': stat['correlation'],
            'Std_Ratio': stat['std_ratio'],
            'Centered_RMS': stat['rms']
        })
    
    for i, stat in enumerate(test_stats):
        taylor_results.append({
            'Model': stat['name'],
            'Dataset': 'Test',
            'Correlation': stat['correlation'],
            'Std_Ratio': stat['std_ratio'],
            'Centered_RMS': stat['rms']
        })
    
    taylor_df = pd.DataFrame(taylor_results)
    taylor_df.to_csv('taylor_statistics.csv', index=False)
    
    print("\nTaylor图表分析完成！")
    print("生成的文件:")
    print("- taylor_diagram.png: Taylor图表（极坐标）")
    print("- taylor_statistics_scatter.png: Taylor统计散点图")
    print("- taylor_statistics.csv: Taylor统计数据")
    
    print("\nTaylor统计结果:")
    print(taylor_df.to_string(index=False))

if __name__ == "__main__":
    create_taylor_diagram()
