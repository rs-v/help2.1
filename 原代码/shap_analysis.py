

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP库可用")
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP库不可用，请安装: pip install shap")

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False


def _set_axis_tick_style(ax, size: int = 18, weight: str = "bold"):
    ax.tick_params(axis='both', labelsize=size)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight(weight)

def perform_shap_analysis():
    """执行SHAP可解释性分析"""
    
    if not SHAP_AVAILABLE:
        print("SHAP库未安装，无法进行分析")
        return
    
    # 1. 加载数据
    print("加载数据...")
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
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 2. 训练最佳模型（XGBoost）
    print("训练XGBoost模型...")
    model = XGBRegressor(
        n_estimators=100, 
        max_depth=7, 
        learning_rate=0.1, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 3. SHAP分析
    print("开始SHAP分析...")
    
    # 选择样本进行分析（减少计算时间）
    sample_size = min(100, len(X_test))
    sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test.iloc[sample_indices]
    
    try:
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        print("SHAP值计算完成")
        
        # 4. 生成SHAP图表
        # (1) SHAP摘要图
        print("生成SHAP摘要图...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, 
                        feature_names=X.columns,
                        show=False)
        plt.title('XGBoost Model - SHAP Feature Importance Summary', 
                 fontsize=14, pad=20, fontfamily='Times New Roman')
        plt.tight_layout()
        plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        # (2) SHAP特征重要性条形图
        print("生成SHAP特征重要性条形图...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, 
                          feature_names=X.columns,
                          plot_type="bar", 
                          show=False)
        plt.title('XGBoost Model - SHAP Feature Importance Bar Chart', 
                  fontsize=14, pad=20, fontfamily='Times New Roman')
        plt.xticks(fontsize=18, fontweight='bold')
        plt.yticks(fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig('shap_importance_bar.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # (3) SHAP依赖图（针对最重要的特征）
        print("生成SHAP依赖图...")
        
        # 计算特征重要性
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        top_features_idx = np.argsort(feature_importance)[-4:]  # 前4个重要特征
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, feature_idx in enumerate(top_features_idx):
            feature_name = X.columns[feature_idx]
            
            # 创建依赖图
            shap.dependence_plot(
                feature_idx, 
                shap_values, 
                X_sample,
                feature_names=X.columns,
                ax=axes[i], 
                show=False
            )
            axes[i].set_title(f'{feature_name} SHAP Dependence', fontsize=12, fontfamily='Times New Roman')        
        plt.suptitle('XGBoost Model - Key Features SHAP Dependence Plots', 
                    fontsize=16, fontfamily='Times New Roman')
        plt.tight_layout()
        plt.savefig('shap_dependence_plots.png', dpi=300, bbox_inches='tight')
        plt.show()          # (4) SHAP瀑布图（单个预测示例）
        print("生成SHAP瀑布图...")
        try:
            plt.figure(figsize=(12, 8))
            # 创建解释对象用于瀑布图
            explanation = shap.Explanation(
                values=shap_values[0], 
                base_values=explainer.expected_value, 
                data=X_sample.iloc[0].values,
                feature_names=X.columns.tolist()
            )
            shap.waterfall_plot(explanation, show=False)
            plt.title('XGBoost Model - Individual Prediction SHAP Waterfall', 
                     fontsize=14, pad=20, fontfamily='Times New Roman')
            plt.tight_layout()
            plt.savefig('shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"瀑布图生成失败: {e}")
          # (5) SHAP力图（显示单个预测的解释）
        print("生成SHAP力图...")
        try:
            plt.figure(figsize=(12, 3))
            shap.force_plot(
                explainer.expected_value, 
                shap_values[0], 
                X_sample.iloc[0],
                feature_names=X.columns,
                matplotlib=True,
                show=False
            )
            plt.title('XGBoost Model - Individual Prediction SHAP Force Plot', 
                     fontsize=14, fontfamily='Times New Roman')
            plt.tight_layout()
            plt.savefig('shap_force_plot.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"力图生成失败: {e}")
        
        # 6. 保存SHAP数值结果
        print("保存SHAP数值结果...")
        
        # 创建SHAP值DataFrame
        shap_df = pd.DataFrame(
            shap_values, 
            columns=[f'SHAP_{col}' for col in X.columns]
        )
        
        # 添加原始特征值
        for i, col in enumerate(X.columns):
            shap_df[f'Feature_{col}'] = X_sample.iloc[:, i].values
        
        # 添加预测值
        predictions = model.predict(X_sample)
        shap_df['Prediction'] = predictions
        shap_df['Actual'] = y_test.iloc[sample_indices].values
        
        # 保存到CSV
        shap_df.to_csv('shap_analysis_results.csv', index=False)
        
        # 7. 创建特征重要性总结
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'SHAP_Importance': feature_importance
        }).sort_values('SHAP_Importance', ascending=False)
        
        feature_importance_df.to_csv('shap_feature_importance.csv', index=False)
        
        print("\nSHAP分析完成！")
        print("生成的文件:")
        print("- shap_summary_plot.png: SHAP摘要图")
        print("- shap_importance_bar.png: SHAP重要性条形图")
        print("- shap_dependence_plots.png: SHAP依赖图")
        print("- shap_waterfall_plot.png: SHAP瀑布图")
        print("- shap_force_plot.png: SHAP力图")
        print("- shap_analysis_results.csv: SHAP数值结果")
        print("- shap_feature_importance.csv: 特征重要性排序")
        
        # 8. 打印特征重要性排序
        print("\n特征重要性排序（基于SHAP值）:")
        print(feature_importance_df.to_string(index=False))
        
    except Exception as e:
        print(f"SHAP分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    perform_shap_analysis()
