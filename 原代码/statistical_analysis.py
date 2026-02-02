import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
import warnings
warnings.filterwarnings('ignore')

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

def calculate_statistics(actual, predicted):
    """计算各种统计指标"""
    
    # 基本统计量
    n = len(actual)
    
    # 1. R² (决定系数)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # 2. WI (Willmott's Index of Agreement)
    numerator = np.sum((predicted - actual) ** 2)
    denominator = np.sum((np.abs(predicted - np.mean(actual)) + np.abs(actual - np.mean(actual))) ** 2)
    wi = 1 - (numerator / denominator)
    
    # 3. PI (Performance Index) - 这里使用常见的定义
    pi = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
    
    # 4. NSE (Nash-Sutcliffe Efficiency)
    nse = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
    
    # 5. RMSE (Root Mean Square Error)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    # 6. MAE (Mean Absolute Error)
    mae = mean_absolute_error(actual, predicted)
    
    # 7. MSE (Mean Square Error)
    mse = mean_squared_error(actual, predicted)
    
    # 8. RMDR (Root Mean Squared Relative Error)
    rmdr = np.sqrt(np.mean(((actual - predicted) / actual) ** 2))
    
    # 9. WMAPE (Weighted Mean Absolute Percentage Error)
    wmape = np.sum(np.abs(actual - predicted)) / np.sum(actual) * 100
    
    # 10. EVS (Explained Variance Score)
    evs = explained_variance_score(actual, predicted)
    
    # 11. RSR (Root Mean Square Error-observations Standard deviation Ratio)
    rsr = rmse / np.std(actual)
    
    return {
        'R²': r2,
        'WI': wi,
        'PI': pi,
        'NSE': nse,
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse,
        'RMDR': rmdr,
        'WMAPE': wmape,
        'EVS': evs,
        'RSR': rsr
    }

def plot_residual_analysis(actual, predicted, save_path='residual_analysis.png'):
    """绘制残差分析图"""
    
    # 计算残差
    residuals = actual - predicted
      # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 残差密度分布
    hist_values, bin_edges = np.histogram(residuals, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    axes[0, 0].hist(residuals, bins=50, density=True, alpha=0.7, color='skyblue', 
                    label='Residual Distribution', edgecolor='black')
    
    # 叠加正态分布
    mu, sigma = stats.norm.fit(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    normal_dist = stats.norm.pdf(x, mu, sigma)
    
    # 保存密度分布数据到CSV
    density_data = pd.DataFrame({
        'bin_center': bin_centers,
        'residual_density': hist_values
    })
      # 创建对应bin_center的正态分布密度值
    normal_density_at_bins = stats.norm.pdf(bin_centers, mu, sigma)
    density_data['normal_density'] = normal_density_at_bins
    
    # 保存更详细的正态分布数据
    normal_data = pd.DataFrame({
        'x_value': x,
        'normal_density': normal_dist
    })
    
    density_data.to_csv('residual_density_distribution.csv', index=False)
    normal_data.to_csv('normal_density_distribution.csv', index=False)
    
    axes[0, 0].plot(x, normal_dist, 'r-', linewidth=2, label=f'Normal Distribution (μ={mu:.4f}, σ={sigma:.4f})')
    
    axes[0, 0].set_xlabel('Residual Value', fontfamily='Times New Roman')
    axes[0, 0].set_ylabel('Density', fontfamily='Times New Roman')
    axes[0, 0].set_title('Residual Density Distribution vs Normal Distribution', fontfamily='Times New Roman')
    axes[0, 0].legend(prop={'family': 'Times New Roman'})
    axes[0, 0].grid(True, alpha=0.3)
      # 2. Q-Q图 (正态性检验)
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_xlabel('Theoretical Quantiles', fontfamily='Times New Roman')
    axes[0, 1].set_ylabel('Sample Quantiles', fontfamily='Times New Roman')
    axes[0, 1].set_title('Q-Q Plot (Normality Test)', fontfamily='Times New Roman')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 残差 vs 预测值散点图
    axes[1, 0].scatter(predicted, residuals, alpha=0.6, color='blue', s=20)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Predicted Values', fontfamily='Times New Roman')
    axes[1, 0].set_ylabel('Residuals', fontfamily='Times New Roman')
    axes[1, 0].set_title('Residuals vs Predicted Values', fontfamily='Times New Roman')
    axes[1, 0].grid(True, alpha=0.3)
      # 4. 实际值 vs 预测值散点图
    axes[1, 1].scatter(actual, predicted, alpha=0.6, color='green', s=20)
    
    # 添加1:1线
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')
    
    axes[1, 1].set_xlabel('Actual Values', fontfamily='Times New Roman')
    axes[1, 1].set_ylabel('Predicted Values', fontfamily='Times New Roman')
    axes[1, 1].set_title('Actual vs Predicted Values', fontfamily='Times New Roman')
    axes[1, 1].legend(prop={'family': 'Times New Roman'})
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 正态性统计检验
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(mu, sigma))
    
    return {
        'residuals_mean': np.mean(residuals),
        'residuals_std': np.std(residuals),
        'shapiro_stat': shapiro_stat,
        'shapiro_p_value': shapiro_p,
        'ks_stat': ks_stat,
        'ks_p_value': ks_p,
        'normal_mu': mu,
        'normal_sigma': sigma
    }

def main():
    """主函数"""
    print("=" * 60)
    print("                统计分析报告")
    print("=" * 60)
    
    # 读取数据
    try:
        df = pd.read_csv('statistic.csv')
        print(f"数据加载成功！数据点数量: {len(df)}")
        print(f"数据列名: {list(df.columns)}")
        print()
        
        actual = df['actual'].values
        predicted = df['predict'].values
        
        # 数据基本信息
        print("数据基本统计信息:")
        print("-" * 40)
        print(f"实际值范围: [{actual.min():.4f}, {actual.max():.4f}]")
        print(f"预测值范围: [{predicted.min():.4f}, {predicted.max():.4f}]")
        print(f"实际值均值: {actual.mean():.4f}")
        print(f"预测值均值: {predicted.mean():.4f}")
        print()
        
        # 计算统计指标
        stats_results = calculate_statistics(actual, predicted)
        
        print("统计指标计算结果:")
        print("-" * 40)
        for metric, value in stats_results.items():
            if metric == 'WMAPE':
                print(f"{metric:8s}: {value:8.4f}%")
            else:
                print(f"{metric:8s}: {value:8.4f}")
        print()
        
        # 残差分析
        print("正在生成残差分析图...")
        residual_results = plot_residual_analysis(actual, predicted)
        
        print("残差分析结果:")
        print("-" * 40)
        print(f"残差均值: {residual_results['residuals_mean']:8.6f}")
        print(f"残差标准差: {residual_results['residuals_std']:8.6f}")
        print(f"拟合正态分布参数: μ={residual_results['normal_mu']:.6f}, σ={residual_results['normal_sigma']:.6f}")
        print()
        print("正态性检验:")
        print(f"Shapiro-Wilk检验: 统计量={residual_results['shapiro_stat']:.6f}, p值={residual_results['shapiro_p_value']:.6f}")
        print(f"Kolmogorov-Smirnov检验: 统计量={residual_results['ks_stat']:.6f}, p值={residual_results['ks_p_value']:.6f}")
        
        if residual_results['shapiro_p_value'] > 0.05:
            print("结论: 残差服从正态分布 (p > 0.05)")
        else:
            print("结论: 残差不服从正态分布 (p ≤ 0.05)")
          # 保存结果到文件
        results_summary = {
            **stats_results,
            **residual_results
        }
        
        results_df = pd.DataFrame([results_summary])
        results_df.to_csv('statistical_results.csv', index=False)
        print()
        print("结果已保存到 'statistical_results.csv' 文件")
        print("残差分析图已保存为 'residual_analysis.png'")
        print("残差密度分布数据已保存到 'residual_density_distribution.csv'")
        print("正态密度分布数据已保存到 'normal_density_distribution.csv'")
        
    except FileNotFoundError:
        print("错误: 未找到 'statistic.csv' 文件")
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()
