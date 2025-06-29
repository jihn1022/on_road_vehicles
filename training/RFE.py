import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from pathlib import Path
import sys
from sklearn.feature_selection import RFECV # 导入 RFECV

# 设置中文字体和忽略警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

class BatteryLifePredictionModel:
    def __init__(self, data_folder=None, num_early_cycles=40):
        """
        初始化电池寿命预测模型
        
        Args:
            data_folder: 数据文件夹路径，如果为None则自动查找
            num_early_cycles: 使用的早期循环数据量
        """
        self.num_early_cycles = num_early_cycles
        self.data_folder = self._find_data_folder(data_folder)
        self.feature_scaler = StandardScaler() # 虽然目前没用，但保留
        self.target_scaler = StandardScaler()
        self.model = None
        self.results = {}
        self.selected_feature_indices = None # 新增：存储RFE选择的特征索引
        self.engineered_feature_names = [] # 新增：存储工程特征名称

    def _find_data_folder(self, data_folder):
        """自动查找数据文件夹"""
        if data_folder and os.path.exists(data_folder):
            return data_folder
            
        # 可能的数据路径
        possible_paths = [
            "./normalized_features",
            "../normalized_features", 
            "./data/normalized_features",
            "../data/normalized_features",
            "/home/jihn/battery-charging-data-of-on-road-electric_vehicles/normalized_features"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                csv_files = glob.glob(os.path.join(path, "*.csv"))
                if csv_files:
                    print(f"找到数据文件夹: {path}")
                    return path
                    
        raise FileNotFoundError("未找到包含CSV文件的数据文件夹，请手动指定data_folder参数")

    def load_csv_data(self):
        """加载CSV数据"""
        csv_files = glob.glob(os.path.join(self.data_folder, "*.csv"))
        csv_files = sorted(csv_files)
        
        if not csv_files:
            raise FileNotFoundError(f"在 {self.data_folder} 中未找到CSV文件")
            
        print(f"找到 {len(csv_files)} 个CSV文件")
        
        early_cycle_data_list = []
        final_cycle_counts = []
        file_names = []
        
        for file_path in csv_files:
            try:
                # 尝试不同的编码方式读取CSV
                df_full = None
                for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
                    try:
                        df_full = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df_full is None:
                    print(f"警告: 无法读取文件 {file_path}")
                    continue
                
                # 检查数据格式
                if df_full.shape[1] < 15:
                    print(f"警告: 文件 {file_path} 列数不足15列，跳过")
                    continue
                
                total_cycles = len(df_full)
                num_cycles_to_read = min(self.num_early_cycles, total_cycles)
                
                # 提取前15列作为特征
                early_features = df_full.iloc[:num_cycles_to_read, :15].values
                
                # 检查数据中是否有无效值
                if np.any(np.isnan(early_features)) or np.any(np.isinf(early_features)):
                    # 填充无效值
                    early_features = np.nan_to_num(early_features, nan=0.0, posinf=1.0, neginf=-1.0)
                
                cycle_numbers = np.arange(1, num_cycles_to_read + 1).reshape(-1, 1)
                early_data_with_cycle = np.hstack((early_features, cycle_numbers))
                
                early_cycle_data_list.append(early_data_with_cycle)
                final_cycle_counts.append(total_cycles)
                file_names.append(os.path.basename(file_path))
                
                print(f"处理完成: {os.path.basename(file_path)} - 早期数据形状: {early_data_with_cycle.shape}, 最终循环数: {total_cycles}")
                
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")
                continue
        
        if not early_cycle_data_list:
            raise ValueError("没有成功加载任何数据文件")
            
        final_cycle_counts = np.array(final_cycle_counts)
        
        print(f"\n数据加载完成!")
        print(f"成功加载 {len(early_cycle_data_list)} 个电池样本的早期循环数据")
        print(f"最终循环数数组形状: {final_cycle_counts.shape}")
        
        return early_cycle_data_list, final_cycle_counts, file_names

    def extract_battery_level_features(self, early_cycle_data_list, file_names):
        """提取电池级别特征"""
        X_engineered = []
        print(f"\n开始特征工程...")
        
        # 生成特征名称列表，用于RFE后的特征重要性可视化
        self.engineered_feature_names = []
        statistic_names = ['Mean', 'Std', 'Min', 'Max', 'First', 'Last', 'Slope']
        num_original_features = 16 # 15个原始特征 + 1个循环数特征
        for i in range(num_original_features):
            original_feature_name = f'Original_F{i}' if i < 15 else 'Cycle_Number'
            for stat_name in statistic_names:
                self.engineered_feature_names.append(f'{original_feature_name}_{stat_name}')

        for i, battery_data in enumerate(early_cycle_data_list):
            engineered_features_for_this_battery = []
            
            for feature_idx in range(battery_data.shape[1]):
                feature_series = battery_data[:, feature_idx]
                
                if len(feature_series) > 0:
                    # 基本统计特征
                    engineered_features_for_this_battery.append(np.mean(feature_series))
                    engineered_features_for_this_battery.append(np.std(feature_series) if len(feature_series) > 1 else 0.0)
                    engineered_features_for_this_battery.append(np.min(feature_series))
                    engineered_features_for_this_battery.append(np.max(feature_series))
                    engineered_features_for_this_battery.append(feature_series[0])
                    engineered_features_for_this_battery.append(feature_series[-1])
                    
                    # 趋势特征
                    if len(feature_series) > 1:
                        if feature_idx == 15:  # Cycle number feature
                            x_values = feature_series
                        else:
                            x_values = np.arange(1, len(feature_series) + 1)
                        
                        try:
                            slope, _, _, _, _ = linregress(x_values, feature_series)
                            if np.isnan(slope) or np.isinf(slope):
                                slope = 0.0
                        except:
                            slope = 0.0
                        
                        engineered_features_for_this_battery.append(slope)
                    else:
                        engineered_features_for_this_battery.extend([0.0] * 7)
                else:
                    engineered_features_for_this_battery.extend([0.0] * 7)
            
            X_engineered.append(engineered_features_for_this_battery)
            
            if (i + 1) % 10 == 0:
                print(f"  已处理 {i + 1}/{len(early_cycle_data_list)} 个电池样本")
        
        X_engineered_array = np.array(X_engineered)
        
        # 检查特征数组中的无效值
        if np.any(np.isnan(X_engineered_array)) or np.any(np.isinf(X_engineered_array)):
            X_engineered_array = np.nan_to_num(X_engineered_array, nan=0.0, posinf=1.0, neginf=-1.0)
            print("警告: 检测到并修复了特征数组中的无效值")
        
        print(f"特征工程完成! 最终工程特征形状: {X_engineered_array.shape}")
        return X_engineered_array

    def train_and_evaluate_model_loocv(self, X_engineered, final_cycle_counts):
        """使用留一法交叉验证训练和评估模型"""
        print(f"\n开始模型训练和评估 (LOOCV)...")
        print(f"输入特征形状: {X_engineered.shape}, 目标变量形状: {final_cycle_counts.shape}")

        X_raw = X_engineered
        print(f"使用原始特征，无标准化. X的均值和标准差: {np.mean(X_raw):.2f}, {np.std(X_raw):.2f}")

        y_log1p = np.log1p(final_cycle_counts)  # log1p变换
        print(f"目标变量log1p变换完成. 变换后Y的范围: [{np.min(y_log1p):.4f}, {np.max(y_log1p):.4f}]")
        
        y_scaled = self.target_scaler.fit_transform(y_log1p.reshape(-1, 1)).flatten()
        print(f"目标变量标准化完成. 标准化后Y的均值和标准差: {np.mean(y_scaled):.2f}, {np.std(y_scaled):.2f}")
        
        # 定义基础模型，RFE将用它来评估特征重要性
        base_estimator = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=1.0, # 这里的max_features是RFE内部每次训练时RF使用的特征比例
            bootstrap=True,
            oob_score=False, # RFE/RFECV内部进行CV，oob_score在这里不直接使用
            warm_start=False, # RFE/RFECV内部会多次训练，这里设置为False
            random_state=123,
            n_jobs=-1,
            min_weight_fraction_leaf=0.0,
        )

        # 创建RFECV实例
        # scoring='neg_mean_squared_error' 是回归任务常用的评分，RFECV会尝试最大化这个值
        # cv=loo 意味着RFECV在内部进行特征选择时也使用留一法
        print("\n开始RFECV特征选择...")
        rfe_selector = RFECV(
            estimator=base_estimator,
            step=1, # 每次消除一个特征
            cv=LeaveOneOut(), # 使用LOOCV进行特征选择
            scoring='neg_mean_squared_error', # 评估标准
            n_jobs=-1 # 并行处理
        )
        rfe_selector.fit(X_raw, y_scaled) # 在所有数据上进行特征选择

        self.selected_feature_indices = rfe_selector.support_
        selected_feature_count = self.selected_feature_indices.sum()
        print(f"RFECV选择的最佳特征数量: {selected_feature_count}")
        print(f"最佳特征的排名（1表示选中）: {rfe_selector.ranking_}")

        # 过滤特征，只保留被RFECV选中的特征
        X_selected = X_raw[:, self.selected_feature_indices]
        print(f"使用RFECV选择的特征进行训练，新特征形状: {X_selected.shape}")

        # 创建最终的随机森林模型，现在它将只在选定的特征上训练
        self.model = RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=1.0, # 这里的max_features是RF在选定特征子集内部的比例
            warm_start=True,
            random_state=123,
            n_jobs=-1,
            min_weight_fraction_leaf=0.0,
        )

        # 留一法交叉验证 (现在在选定的特征上进行)
        loo = LeaveOneOut()
        y_true_all = []
        y_pred_all = []
        
        total_folds = len(X_selected) # 注意这里是X_selected
        for i, (train_index, test_index) in enumerate(loo.split(X_selected)): # 注意这里是X_selected
            X_train, X_test = X_selected[train_index], X_selected[test_index]
            y_train, y_test = y_scaled[train_index], y_scaled[test_index]

            self.model.fit(X_train, y_train)
            y_pred_fold_scaled = self.model.predict(X_test)
            
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred_fold_scaled)
            
            if (i + 1) % 10 == 0:
                print(f"  已完成 {i + 1}/{total_folds} 折交叉验证")
    
        # 转换回原始尺度
        y_true_log1p = self.target_scaler.inverse_transform(np.array(y_true_all).reshape(-1, 1)).flatten()
        y_pred_log1p = self.target_scaler.inverse_transform(np.array(y_pred_all).reshape(-1, 1)).flatten()
        
        y_true_original = np.expm1(y_true_log1p)
        y_pred_original = np.expm1(y_pred_log1p)
        
        y_pred_original = np.maximum(y_pred_original, 0)

        # 计算评估指标
        overall_rmse = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
        overall_mae = mean_absolute_error(y_true_original, y_pred_original)
        overall_r2 = r2_score(y_true_original, y_pred_original)

        print(f"\n{'='*60}")
        print(f"LOOCV 随机森林模型评估结果 (原始尺度，使用log1p变换，经过RFECV特征选择)")
        print(f"{'='*60}")
        print(f"  RMSE: {overall_rmse:.4f}")
        print(f"  MAE:  {overall_mae:.4f}")
        print(f"  R² Score: {overall_r2:.4f}")
        
        print(f"\n预测示例 (前10个样本):")
        print(f"{'实际值':<10} {'预测值':<10} {'误差':<10} {'误差率%':<10}")
        print("-" * 50)
        for i in range(min(10, len(y_true_original))):
            actual = y_true_original[i]
            predicted = y_pred_original[i]
            error = abs(actual - predicted)
            error_rate = (error / actual * 100) if actual != 0 else np.inf
            print(f"{actual:<10.2f} {predicted:<10.2f} {error:<10.2f} {error_rate:<10.2f}%")
        
        # 重新训练模型用于后续预测 (在所有数据和选定特征上训练最终模型)
        self.model.fit(X_selected, y_scaled) # 注意这里是X_selected
        
        self.results = {
            'model': self.model,
            'feature_scaler': None,
            'target_scaler': self.target_scaler,
            'overall_rmse': overall_rmse,
            'overall_mae': overall_mae,
            'overall_r2': overall_r2,
            'y_true_original': y_true_original,
            'y_pred_original': y_pred_original,
            'X_raw': X_raw, # 原始工程特征，用于保存和参考
            'X_selected': X_selected, # 选定特征
            'selected_feature_indices': self.selected_feature_indices, # 选定特征的索引
            'feature_importances': self.model.feature_importances_, # 这是在选定特征子集上的重要性
            'use_log1p': True
        }
        
        return self.results

    def visualize_feature_importance(self, n_top=20):
        """可视化特征重要性 (现在考虑RFE选择后的特征)"""
        if not self.results or self.selected_feature_indices is None:
            print("请先训练模型或RFE未执行")
            return
            
        feature_importances = self.results['feature_importances']
        
        # 获取被RFECV选中的特征名称
        selected_engineered_feature_names = np.array(self.engineered_feature_names)[self.selected_feature_indices]

        feature_df = pd.DataFrame({
            'feature_name': selected_engineered_feature_names,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
        
        top_features = feature_df.head(n_top)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature_name', data=top_features, palette='viridis')
        plt.xlabel('特征重要性')
        plt.ylabel('工程特征名称')
        plt.title(f'前 {n_top} 个工程特征重要性 (RFECV选择后)')
        plt.tight_layout()
        
        try:
            plt.savefig('feature_importance_rfe.png', dpi=300, bbox_inches='tight')
            print("特征重要性图已保存为 feature_importance_rfe.png")
        except:
            print("保存图片失败，但显示正常")
        
        plt.show()

    def save_results_to_file(self, file_names, output_file='model_evaluation_results_rfe.txt'): # 修改文件名
        """保存评估结果到文件"""
        if not self.results:
            print("请先训练模型")
            return
            
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("随机森林模型评估报告 (带RFECV特征选择)\n") # 修改标题
            f.write("=" * 50 + "\n\n")
            
            f.write(f"数据集信息:\n")
            f.write(f"  电池样本数量: {len(self.results['y_true_original'])}\n")
            f.write(f"  初始工程特征数量: {self.results['X_raw'].shape[1]}\n")
            f.write(f"  RFECV选择的特征数量: {self.results['X_selected'].shape[1]}\n\n") # 新增
            
            f.write(f"模型性能指标 (LOOCV, 原始尺度):\n")
            f.write(f"  RMSE: {self.results['overall_rmse']:.4f}\n")
            f.write(f"  MAE:  {self.results['overall_mae']:.4f}\n")
            f.write(f"  R²:   {self.results['overall_r2']:.4f}\n\n")
            
            f.write(f"详细预测结果:\n")
            f.write(f"{'文件名':<30} {'实际值':<10} {'预测值':<10} {'绝对误差':<10} {'误差率%':<10}\n")
            f.write("-" * 75 + "\n")
            
            for i in range(len(self.results['y_true_original'])):
                actual = self.results['y_true_original'][i]
                predicted = self.results['y_pred_original'][i]
                error = abs(actual - predicted)
                error_rate = (error / actual * 100) if actual != 0 else np.inf
                f.write(f"{file_names[i]:<30} {actual:<10.2f} {predicted:<10.2f} {error:<10.2f} {error_rate:<10.2f}\n")

    def save_model(self, model_dir='models'):
        """保存模型和预处理器"""
        if not self.results:
            print("请先训练模型")
            return
            
        Path(model_dir).mkdir(exist_ok=True)
        
        # 保存模型和目标变量标准化器
        with open(f'{model_dir}/final_rf_model_rfe.pkl', 'wb') as f: # 修改文件名
            pickle.dump(self.model, f)
        with open(f'{model_dir}/target_scaler_rfe.pkl', 'wb') as f: # 修改文件名
            pickle.dump(self.target_scaler, f)
        # 保存选择的特征索引，以便预测时使用
        with open(f'{model_dir}/selected_feature_indices.pkl', 'wb') as f:
            pickle.dump(self.selected_feature_indices, f)
    
        print(f"\n模型和预处理器已保存到 {model_dir}/ 目录")
        print("保存的文件包括:")
        print(f"- {model_dir}/final_rf_model_rfe.pkl: 训练好的随机森林模型 (RFE后)")
        print(f"- {model_dir}/target_scaler_rfe.pkl: 目标变量标准化器")
        print(f"- {model_dir}/selected_feature_indices.pkl: RFE选择的特征索引")


    def predict(self, X_new):
        """对新数据进行预测"""
        if self.model is None or self.selected_feature_indices is None:
            raise ValueError("模型尚未训练或RFE未执行，请先调用训练方法")
        
        # 对新数据也应用相同的特征选择
        X_new_selected = X_new[:, self.selected_feature_indices]
    
        y_pred_scaled = self.model.predict(X_new_selected)
        
        y_pred_log1p = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        y_pred_original = np.expm1(y_pred_log1p)
        
        y_pred_original = np.maximum(y_pred_original, 0)
        
        return y_pred_original

    def plot_prediction_results(self):
        """绘制预测结果"""
        if not self.results:
            print("请先训练模型")
            return
            
        y_true = self.results['y_true_original']
        y_pred = self.results['y_pred_original']
        r2 = self.results['overall_r2']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 实际值 vs 预测值
        axes[0].scatter(y_true, y_pred, alpha=0.7, color='blue')
        max_val = max(y_true.max(), y_pred.max())
        min_val = min(y_true.min(), y_pred.min())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0].set_xlabel('实际循环数')
        axes[0].set_ylabel('预测循环数')
        axes[0].set_title(f'LOOCV 实际值 vs 预测值 (R² = {r2:.4f})')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal', adjustable='box')

        # 残差图
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.7, color='green')
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('预测循环数')
        axes[1].set_ylabel('残差 (实际值 - 预测值)')
        axes[1].set_title('LOOCV 残差图')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        try:
            plt.savefig('prediction_results_rfe.png', dpi=300, bbox_inches='tight')
            print("预测结果图已保存为 prediction_results_rfe.png")
        except:
            print("保存图片失败，但显示正常")
        
        plt.show()

    def run_complete_pipeline(self):
        """运行完整的预测流水线"""
        try:
            # 1. 加载数据
            early_data_list, final_cycle_counts, file_names = self.load_csv_data()
            
            # 2. 特征工程
            X_engineered = self.extract_battery_level_features(early_data_list, file_names)
            
            # 3. 训练和评估模型 (包含RFECV特征选择)
            self.train_and_evaluate_model_loocv(X_engineered, final_cycle_counts)
            
            # 4. 可视化结果
            self.plot_prediction_results()
            self.visualize_feature_importance()
            
            # 5. 保存结果
            self.save_results_to_file(file_names)
            self.save_model()
            
            print("\n✅ 完整的预测流水线执行完成!")
            print(f"使用RFECV选择了 {self.selected_feature_indices.sum()} 个最佳特征")
            
        except Exception as e:
            print(f"❌ 执行过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    print("🔋 电池寿命预测模型")
    print("=" * 50)
    
    # 创建模型实例
    model = BatteryLifePredictionModel(
        data_folder=None,  # 自动查找数据文件夹
        num_early_cycles=40
    )
    
    # 运行完整流水线
    model.run_complete_pipeline()

if __name__ == "__main__":
    main()