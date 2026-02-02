

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine learning model imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Visualization and analysis tools
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
from scipy import stats
import joblib
from datetime import datetime

# Set Times New Roman font for all plots
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

class SaltCavernMLAnalysis:
    """Salt Cavern Gas Storage Volume Shrinkage Prediction Machine Learning Analysis Class"""
    
    def __init__(self, data_path):
        """Initialize analysis class"""
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.predictions = {}
        self.metrics = {}
          # Feature name mapping
        self.feature_names = {
            'S/L': 'S/L Ratio',
            'CD': 'Cavern Depth (m)', 
            'CV': 'Cavern Volume (m³)',
            'PW': 'Pillar Width Ratio',
            'LP': 'Low Pressure (MPa)',
            'HP': 'High Pressure (MPa)', 
            'F': 'Frequency (times/year)',
            'E': 'Elastic Modulus (GPa)',
            'v': 'Poisson Ratio',
            'c': 'Cohesion (MPa)',
            'φ': 'Friction Angle (°)',
            'T': 'Tensile Strength (MPa)',
            'A': 'Creep Parameter A',
            'n': 'Creep Index n',
            'VS': 'Volume Shrinkage Rate (%)'
        }
        
    def load_and_preprocess_data(self):
        """Load and preprocess data"""
        print("Loading and preprocessing data...")
        
        # Read data
        self.data = pd.read_csv(self.data_path)        # Remove last column if empty
        if self.data.columns[-1] == '' or 'Unnamed' in str(self.data.columns[-1]):
            print(f"Removing empty column: {self.data.columns[-1]}")
            self.data = self.data.drop(self.data.columns[-1], axis=1)
        
        print(f"Data shape: {self.data.shape}")
        print(f"Column names: {list(self.data.columns)}")
        
        # Basic data information
        print("\nBasic data information:")
        print(self.data.info())
        print("\nMissing value statistics:")
        print(self.data.isnull().sum())
        # Handle missing values - fill numeric features with median
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if self.data[col].isnull().any():
                # Only fill if there are non-null values to calculate median from
                non_null_values = self.data[col].dropna()
                if len(non_null_values) > 0:
                    median_val = non_null_values.median()
                    self.data[col].fillna(median_val, inplace=True)
                    print(f"Feature {col} filled missing values with median {median_val:.4f}")
                else:
                    print(f"Feature {col} is entirely NaN, dropping this column")
                    self.data = self.data.drop(col, axis=1)
          # Separate features and target variable
        feature_columns = [col for col in self.data.columns if col != 'VS']
        self.X = self.data[feature_columns]
        self.y = self.data['VS']
        
        print(f"\nNumber of features: {self.X.shape[1]}")
        print(f"Number of samples: {self.X.shape[0]}")
        print(f"Target variable statistics:\n{self.y.describe()}")
        
        return self.data
        
    def split_and_scale_data(self, test_size=0.2, random_state=42):
        """Data splitting and standardization"""
        print("Splitting and standardizing data...")
        
        # Data splitting
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Feature standardization
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        
    def create_models(self):
        """Create all machine learning models"""
        print("Creating machine learning models...")
        
        # 1. Linear Regression
        self.models['Linear Regression'] = LinearRegression()
        
        # 2. Decision Tree
        self.models['Decision Tree'] = DecisionTreeRegressor(
            max_depth=10, min_samples_split=5, random_state=42
        )
        
        # 3. Random Forest
        self.models['Random Forest'] = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )
        
        # 4. Support Vector Machine
        self.models['SVM'] = SVR(kernel='rbf', C=100, gamma='scale')
        
        # 5. XGBoost
        self.models['XGBoost'] = XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        
        # 6. LightGBM
        self.models['LightGBM'] = LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1
        )
        
        # 7. Gradient Boosting
        self.models['GBM'] = GradientBoostingRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        
        # 8. K-Nearest Neighbors
        self.models['KNN'] = KNeighborsRegressor(n_neighbors=5)
        
        # 9. Gaussian Process Regression
        self.models['Gaussian Process'] = GaussianProcessRegressor(random_state=42)
        
        print(f"Created {len(self.models)} models")
        
    def create_ann_model(self):
        """Create Artificial Neural Network model"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(self.X_train_scaled.shape[1],)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', 
                     metrics=['mae'])
        return model
        
    def train_models(self):
        """Train all models"""
        print("Training models...")
        
        for name, model in self.models.items():
            print(f"Training model: {name}")
            
            if name in ['SVM', 'KNN', 'Gaussian Process']:
                # These models use standardized data
                model.fit(self.X_train_scaled, self.y_train)
                train_pred = model.predict(self.X_train_scaled)
                test_pred = model.predict(self.X_test_scaled)
            else:
                # Other models use original data
                model.fit(self.X_train, self.y_train)
                train_pred = model.predict(self.X_train)
                test_pred = model.predict(self.X_test)
            
            # Save prediction results
            self.predictions[name] = {
                'train': train_pred,
                'test': test_pred
            }
            
            # Calculate evaluation metrics
            self.calculate_metrics(name, train_pred, test_pred)
        
        # Train neural network
        print("Training model: ANN")
        ann_model = self.create_ann_model()
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        
        history = ann_model.fit(
            self.X_train_scaled, self.y_train,
            validation_split=0.2,
            epochs=200,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.models['ANN'] = ann_model
        train_pred = ann_model.predict(self.X_train_scaled).flatten()
        test_pred = ann_model.predict(self.X_test_scaled).flatten()
        
        self.predictions['ANN'] = {
            'train': train_pred,
            'test': test_pred
        }
        
        self.calculate_metrics('ANN', train_pred, test_pred)
        
    def calculate_metrics(self, model_name, train_pred, test_pred):
        """Calculate model evaluation metrics"""
        
        def wmape(y_true, y_pred):
            """Weighted Mean Absolute Percentage Error"""
            return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
        
        # Training set metrics
        train_r2 = r2_score(self.y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
        train_mae = mean_absolute_error(self.y_train, train_pred)
        train_mse = mean_squared_error(self.y_train, train_pred)
        train_wmape = wmape(self.y_train, train_pred)
        
        # Test set metrics
        test_r2 = r2_score(self.y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
        test_mae = mean_absolute_error(self.y_test, test_pred)
        test_mse = mean_squared_error(self.y_test, test_pred)
        test_wmape = wmape(self.y_test, test_pred)
        
        self.metrics[model_name] = {
            'train': {
                'R²': train_r2,
                'RMSE': train_rmse,
                'MAE': train_mae,
                'MSE': train_mse,
                'WMAPE': train_wmape
            },
            'test': {
                'R²': test_r2,
                'RMSE': test_rmse,
                'MAE': test_mae,
                'MSE': test_mse,
                'WMAPE': test_wmape
            }
        }
        
    def plot_correlation_heatmap(self):
        """Generate heatmap showing feature correlations"""
        print("Generating correlation heatmap...")
        
        # Calculate correlation matrix
        correlation_matrix = self.data.corr()
        
        # Create heatmap
        plt.figure(figsize=(16, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": .8},
                   fmt='.3f')
        
        plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create correlation bar plot with target variable VS
        plt.figure(figsize=(12, 8))
        vs_correlation = correlation_matrix['VS'].drop('VS').sort_values(key=abs, ascending=False)
        
        colors = ['red' if x < 0 else 'blue' for x in vs_correlation.values]
        bars = plt.bar(range(len(vs_correlation)), vs_correlation.values, color=colors, alpha=0.7)
        
        plt.xlabel('Features')
        plt.ylabel('Correlation with VS')
        plt.title('Feature Correlation with Volume Shrinkage (VS)')
        plt.xticks(range(len(vs_correlation)), vs_correlation.index, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
          # Add value labels
        for bar, value in zip(bars, vs_correlation.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * np.sign(value), 
                    f'{value:.3f}', ha='center', va='bottom' if value > 0 else 'top')
        
        plt.tight_layout()
        plt.savefig('vs_correlation_barplot.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_accuracy(self):
        """3. Plot predicted vs actual values"""
        print("Generating prediction accuracy visualization...")
        
        # Create subplots
        n_models = len(self.predictions)
        cols = 4
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, (model_name, preds) in enumerate(self.predictions.items()):
            ax = axes[idx]
            
            # Plot training set
            ax.scatter(self.y_train, preds['train'], alpha=0.6, label='Training Set', s=30)
            # Plot test set
            ax.scatter(self.y_test, preds['test'], alpha=0.6, label='Test Set', s=30)
            
            # Plot ideal prediction line
            min_val = min(self.y.min(), min(preds['train'].min(), preds['test'].min()))
            max_val = max(self.y.max(), max(preds['train'].max(), preds['test'].max()))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            # Add regression lines
            z_train = np.polyfit(self.y_train, preds['train'], 1)
            p_train = np.poly1d(z_train)
            ax.plot(sorted(self.y_train), p_train(sorted(self.y_train)), "b-", alpha=0.8, linewidth=1)
            
            z_test = np.polyfit(self.y_test, preds['test'], 1)
            p_test = np.poly1d(z_test)
            ax.plot(sorted(self.y_test), p_test(sorted(self.y_test)), "g-", alpha=0.8, linewidth=1)
            
            # Add performance metrics
            train_r2 = self.metrics[model_name]['train']['R²']
            test_r2 = self.metrics[model_name]['test']['R²']
            train_rmse = self.metrics[model_name]['train']['RMSE']
            test_rmse = self.metrics[model_name]['test']['RMSE']
            
            ax.text(0.05, 0.95, f'Train R²: {train_r2:.3f}\nTest R²: {test_r2:.3f}\nTrain RMSE: {train_rmse:.3f}\nTest RMSE: {test_rmse:.3f}', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{model_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
          # Hide extra subplots
        for idx in range(len(self.predictions), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Predicted vs Actual Values', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig('prediction_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_residual_analysis(self):
        """4. Create residual distribution curves and violin plots"""
        print("Generating residual distribution analysis...")
        
        # Calculate residuals
        residuals = {}
        for model_name, preds in self.predictions.items():
            residuals[model_name] = {
                'train': self.y_train - preds['train'],
                'test': self.y_test - preds['test']
            }
        
        # 1. Residual distribution curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Training set residual distribution
        for model_name, resid in residuals.items():
            ax1.hist(resid['train'], bins=20, alpha=0.6, label=model_name, density=True)
        ax1.set_xlabel('Residuals')
        ax1.set_ylabel('Density')
        ax1.set_title('Training Set Residual Distribution')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Test set residual distribution
        for model_name, resid in residuals.items():
            ax2.hist(resid['test'], bins=20, alpha=0.6, label=model_name, density=True)
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Density')
        ax2.set_title('Test Set Residual Distribution')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('residual_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Violin plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Prepare data
        train_residual_data = []
        test_residual_data = []
        model_names = []
        
        for model_name, resid in residuals.items():
            train_residual_data.extend(resid['train'])
            test_residual_data.extend(resid['test'])
            model_names.extend([model_name] * len(resid['train']))
        
        # Create DataFrame for seaborn
        train_df = pd.DataFrame({
            'Residuals': list(residuals.values())[0]['train'],
            'Model': list(residuals.keys())[0]
        })
        
        for model_name, resid in list(residuals.items())[1:]:
            temp_df = pd.DataFrame({
                'Residuals': resid['train'],
                'Model': model_name
            })
            train_df = pd.concat([train_df, temp_df], ignore_index=True)
        
        test_df = pd.DataFrame({
            'Residuals': list(residuals.values())[0]['test'],
            'Model': list(residuals.keys())[0]
        })
        
        for model_name, resid in list(residuals.items())[1:]:
            temp_df = pd.DataFrame({
                'Residuals': resid['test'],
                'Model': model_name
            })
            test_df = pd.concat([test_df, temp_df], ignore_index=True)
        
        # Plot violin plots
        sns.violinplot(data=train_df, x='Model', y='Residuals', ax=ax1)
        ax1.set_title('Training Set Residual Violin Plot')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        sns.violinplot(data=test_df, x='Model', y='Residuals', ax=ax2)
        ax2.set_title('Test Set Residual Violin Plot')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('residual_violin_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
          # Save residual data
        residual_df = pd.DataFrame()
        for model_name, resid in residuals.items():
            temp_df = pd.DataFrame({
                'Model': model_name,
                'Train_Residuals': list(resid['train']) + [np.nan] * (len(resid['test']) - len(resid['train'])) if len(resid['test']) > len(resid['train']) else list(resid['train']),
                'Test_Residuals': list(resid['test']) + [np.nan] * (len(resid['train']) - len(resid['test'])) if len(resid['train']) > len(resid['test']) else list(resid['test'])
            })
            residual_df = pd.concat([residual_df, temp_df], ignore_index=True)
        
        residual_df.to_csv('residual_analysis.csv', index=False)
    
    def plot_feature_sensitivity(self):
        """5. Generate feature sensitivity analysis bar charts"""
        print("Generating feature sensitivity analysis...")
        
        # Use best models for feature importance analysis
        feature_importance_models = ['Random Forest', 'XGBoost', 'LightGBM', 'GBM']
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()
        
        for idx, model_name in enumerate(feature_importance_models):
            if model_name in self.models:
                model = self.models[model_name]
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = self.X.columns
                    
                    # Sort
                    indices = np.argsort(importances)[::-1]
                    
                    # Plot
                    ax = axes[idx]
                    bars = ax.bar(range(len(importances)), importances[indices])
                    ax.set_xlabel('Features')
                    ax.set_ylabel('Importance')
                    ax.set_title(f'{model_name} Feature Importance')
                    ax.set_xticks(range(len(importances)))
                    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
                      # Add value labels
                    for bar, imp in zip(bars, importances[indices]):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                               f'{imp:.3f}', ha='center', va='bottom', fontsize=8)
                    
                    ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Feature Sensitivity Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig('feature_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def shap_analysis(self):
        """6. SHAP interpretability analysis"""
        print("Performing SHAP interpretability analysis...")
        
        # Select best model for SHAP analysis
        best_model_name = max(self.metrics.keys(), 
                             key=lambda x: self.metrics[x]['test']['R²'])
        best_model = self.models[best_model_name]
        
        print(f"Using best model for SHAP analysis: {best_model_name}")
        
        # Prepare data (select appropriate sample size to avoid slow computation)
        sample_size = min(100, len(self.X_test))
        sample_indices = np.random.choice(len(self.X_test), sample_size, replace=False)
        
        if best_model_name in ['SVM', 'KNN', 'Gaussian Process']:
            X_sample = self.X_test_scaled[sample_indices]
            X_background = self.X_train_scaled[:100]  # Background data
        else:
            X_sample = self.X_test.iloc[sample_indices]
            X_background = self.X_train.iloc[:100]  # Background data
        
        try:
            # Create SHAP explainer
            if best_model_name == 'ANN':
                explainer = shap.DeepExplainer(best_model, X_background)
            elif best_model_name in ['XGBoost', 'LightGBM']:
                explainer = shap.TreeExplainer(best_model)
            else:
                explainer = shap.KernelExplainer(best_model.predict, X_background)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # 1. SHAP summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, 
                            feature_names=self.X.columns,
                            show=False)
            plt.title(f'{best_model_name} - SHAP Summary Plot')
            plt.tight_layout()
            plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 2. SHAP dependence plots (for important features)
            if len(self.X.columns) > 0:
                # Calculate feature importance (based on mean absolute SHAP values)
                feature_importance = np.mean(np.abs(shap_values), axis=0)
                top_features = np.argsort(feature_importance)[-4:]  # Select top 4 important features
                
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                axes = axes.flatten()
                
                for idx, feature_idx in enumerate(top_features):
                    feature_name = self.X.columns[feature_idx]
                    shap.dependence_plot(feature_idx, shap_values, X_sample,
                                       feature_names=self.X.columns,
                                       ax=axes[idx], show=False)
                    axes[idx].set_title(f'SHAP Dependence Plot - {feature_name}')
                
                plt.suptitle(f'{best_model_name} - SHAP Feature Dependence Plots', fontsize=16)
                plt.tight_layout()
                plt.savefig('shap_dependence_plots.png', dpi=300, bbox_inches='tight')
                plt.show()
            
            # 3. SHAP waterfall plot (single prediction example)
            if hasattr(shap, 'waterfall_plot'):
                plt.figure(figsize=(10, 6))
                shap.waterfall_plot(explainer.expected_value, shap_values[0], 
                                  X_sample.iloc[0] if hasattr(X_sample, 'iloc') else X_sample[0],
                                  feature_names=self.X.columns,
                                  show=False)
                plt.title(f'{best_model_name} - SHAP Waterfall Plot (Single Prediction Example)')
                plt.tight_layout()
                plt.savefig('shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
                plt.show()
            
        except Exception as e:
            print(f"SHAP analysis error: {e}")
            print("Trying simplified SHAP analysis...")
            
            # If above method fails, use simplified approach
            try:
                if best_model_name in ['Random Forest', 'XGBoost', 'LightGBM', 'GBM']:
                    # For tree models, use feature importance as alternative                    if hasattr(best_model, 'feature_importances_'):
                        importances = best_model.feature_importances_
                        feature_names = self.X.columns
                        
                        plt.figure(figsize=(10, 6))
                        indices = np.argsort(importances)[::-1]
                        plt.bar(range(len(importances)), importances[indices])
                        plt.xlabel('Features')
                        plt.ylabel('Importance')
                        plt.title(f'{best_model_name} - Feature Importance (SHAP Alternative Analysis)')
                        plt.xticks(range(len(importances)), 
                                  [feature_names[i] for i in indices], rotation=45, ha='right')
                        plt.tight_layout()
                        plt.savefig('feature_importance_alternative.png', dpi=300, bbox_inches='tight')
                        plt.show()
            except Exception as e2:
                print(f"Simplified SHAP analysis also failed: {e2}")
    
    def create_taylor_diagram(self):
        """7. Create Taylor diagram comparison"""
        print("Generating Taylor diagram...")
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Calculate statistics required for Taylor diagram
            def taylor_statistics(reference, model):
                # Standard deviation
                ref_std = np.std(reference)
                mod_std = np.std(model)
                
                # Correlation coefficient
                correlation = np.corrcoef(reference, model)[0, 1]
                
                # Centered root mean square difference
                ref_mean = np.mean(reference)
                mod_mean = np.mean(model)
                centered_rms = np.sqrt(np.mean((model - mod_mean - (reference - ref_mean))**2))
                
                return ref_std, mod_std, correlation, centered_rms
            
            # Create Taylor diagram for training and test sets
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Training Set', 'Test Set'),
                specs=[[{"type": "scatterpolar"}, {"type": "scatterpolar"}]]
            )
            
            # Training set data
            train_stats = []
            test_stats = []
            
            for model_name, preds in self.predictions.items():
                # Training set statistics
                ref_std_train, mod_std_train, corr_train, _ = taylor_statistics(
                    self.y_train, preds['train']
                )
                train_stats.append({
                    'name': model_name,
                    'std': mod_std_train / ref_std_train,  # Normalized standard deviation
                    'correlation': corr_train
                })
                
                # Test set statistics
                ref_std_test, mod_std_test, corr_test, _ = taylor_statistics(
                    self.y_test, preds['test']
                )
                test_stats.append({
                    'name': model_name,
                    'std': mod_std_test / ref_std_test,  # Normalized standard deviation
                    'correlation': corr_test
                })
            
            # Plot training set Taylor diagram
            for stat in train_stats:
                theta = np.arccos(stat['correlation']) * 180 / np.pi  # Convert to degrees
                r = stat['std']
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=[r],
                        theta=[theta],
                        mode='markers+text',
                        text=[stat['name']],
                        textposition="top center",
                        name=stat['name'],
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Plot test set Taylor diagram
            for stat in test_stats:
                theta = np.arccos(stat['correlation']) * 180 / np.pi  # Convert to degrees
                r = stat['std']
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=[r],
                        theta=[theta],
                        mode='markers+text',
                        text=[stat['name']],
                        textposition="top center",
                        name=stat['name'],
                        showlegend=True
                    ),
                    row=1, col=2
                )
            
            # Add reference point (observed values)
            fig.add_trace(
                go.Scatterpolar(
                    r=[1],
                    theta=[0],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name='Reference Observations',
                    showlegend=True
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatterpolar(
                    r=[1],
                    theta=[0],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name='Reference Observations',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title="Taylor Diagram Comparison",
                polar=dict(
                    radialaxis=dict(range=[0, 2], tickangle=0),
                    angularaxis=dict(tickmode='array', 
                                   tickvals=[0, 30, 60, 90, 120, 150, 180],
                                   ticktext=['1.0', '0.87', '0.5', '0.0', '-0.5', '-0.87', '-1.0'])
                ),
                polar2=dict(
                    radialaxis=dict(range=[0, 2], tickangle=0),
                    angularaxis=dict(tickmode='array', 
                                   tickvals=[0, 30, 60, 90, 120, 150, 180],
                                   ticktext=['1.0', '0.87', '0.5', '0.0', '-0.5', '-0.87', '-1.0'])
                )
            )
            
            fig.write_html("taylor_diagram.html")
            fig.show()
            
        except Exception as e:
            print(f"Taylor diagram generation failed: {e}")
            print("Generating simplified version of performance comparison chart...")
            
            # Simplified version: use regular charts
            metrics_df = pd.DataFrame()
            for model_name, metrics in self.metrics.items():
                temp_df = pd.DataFrame({
                    'Model': [model_name, model_name],
                    'Dataset': ['Train', 'Test'],
                    'R²': [metrics['train']['R²'], metrics['test']['R²']],
                    'RMSE': [metrics['train']['RMSE'], metrics['test']['RMSE']]
                })
                metrics_df = pd.concat([metrics_df, temp_df], ignore_index=True)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # R² comparison
            sns.barplot(data=metrics_df, x='Model', y='R²', hue='Dataset', ax=ax1)
            ax1.set_title('Model R² Score Comparison')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis='y', alpha=0.3)
              # RMSE comparison
            sns.barplot(data=metrics_df, x='Model', y='RMSE', hue='Dataset', ax=ax2)
            ax2.set_title('Model RMSE Comparison')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print("Generating comprehensive analysis report...")
        
        # Create performance summary table
        performance_df = pd.DataFrame()
        for model_name, metrics in self.metrics.items():
            temp_df = pd.DataFrame({
                'Model': [model_name],
                'Train_R²': [metrics['train']['R²']],
                'Test_R²': [metrics['test']['R²']],
                'Train_RMSE': [metrics['train']['RMSE']],
                'Test_RMSE': [metrics['test']['RMSE']],
                'Train_MAE': [metrics['train']['MAE']],
                'Test_MAE': [metrics['test']['MAE']],
                'Train_WMAPE': [metrics['train']['WMAPE']],
                'Test_WMAPE': [metrics['test']['WMAPE']]
            })
            performance_df = pd.concat([performance_df, temp_df], ignore_index=True)
        
        # Sort by test set R²
        performance_df = performance_df.sort_values('Test_R²', ascending=False)
        
        # Save results
        performance_df.to_csv('model_performance_summary.csv', index=False)
        
        # Print best model
        best_model = performance_df.iloc[0]
        print(f"\nBest Model: {best_model['Model']}")
        print(f"Test R²: {best_model['Test_R²']:.4f}")
        print(f"Test RMSE: {best_model['Test_RMSE']:.4f}")
        print(f"Test MAE: {best_model['Test_MAE']:.4f}")
        print(f"Test WMAPE: {best_model['Test_WMAPE']:.4f}%")
          # Display performance table
        print("\nAll Model Performance Summary:")
        print(performance_df.to_string(index=False))
        
        return performance_df
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("Starting Salt Cavern Gas Storage Volume Shrinkage Prediction Comprehensive Analysis...")
        print("=" * 60)
        
        # 1. Data loading and preprocessing
        self.load_and_preprocess_data()
        self.split_and_scale_data()
        
        # 2. Create and train models
        self.create_models()
        self.train_models()
        
        # 3. Generate all analysis charts
        self.plot_correlation_heatmap()  # 2. Heatmap
        self.plot_prediction_accuracy()  # 3. Prediction accuracy visualization
        self.plot_residual_analysis()    # 4. Residual distribution analysis
        self.plot_feature_sensitivity()  # 5. Feature sensitivity analysis
        self.shap_analysis()            # 6. SHAP interpretability analysis
        self.create_taylor_diagram()    # 7. Taylor diagram comparison
        
        # 4. Generate comprehensive report
        performance_summary = self.generate_comprehensive_report()
        
        print("\n=" * 60)
        print("Analysis completed! All charts and results have been saved.")
        print("Generated files include:")
        print("- correlation_heatmap.png: Feature correlation heatmap")
        print("- vs_correlation_barplot.png: VS correlation bar chart")
        print("- prediction_accuracy.png: Prediction accuracy chart")
        print("- residual_distribution.png: Residual distribution chart")
        print("- residual_violin_plot.png: Residual violin plot")
        print("- feature_sensitivity.png: Feature sensitivity chart")
        print("- shap_summary_plot.png: SHAP summary plot")
        print("- shap_dependence_plots.png: SHAP dependence plots")
        print("- taylor_diagram.html: Taylor diagram")
        print("- model_performance_summary.csv: Model performance summary")
        print("- residual_analysis.csv: Residual analysis data")
        
        return performance_summary

# Main execution function
def main():
    """Main function"""
    # Create analysis instance
    analyzer = SaltCavernMLAnalysis('dateset.csv')
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    return analyzer, results

if __name__ == "__main__":
    analyzer, results = main()
