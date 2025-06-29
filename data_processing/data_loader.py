import os
import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Any
import re
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, directory: str):
        """
        初始化数据处理器
        
        Args:
            directory: 包含CSV文件的根目录
        """
        self.directory = directory
        self.feature_extractors = {}
        self.processed_files = []
    
    def extract_datetime_from_path(self, file_path: str) -> datetime:
        """
        从文件路径中提取日期时间
        
        Args:
            file_path: 文件路径
            
        Returns:
            datetime对象，如果无法提取则返回一个默认的最早时间
        """
        try:
            # 匹配模式：YYYYMMDD_HHMMSS
            pattern = r'(\d{8})_(\d{6})'
            match = re.search(pattern, file_path)
            
            if match:
                date_str = match.group(1)  # YYYYMMDD
                time_str = match.group(2)  # HHMMSS
                
                # 转换为datetime对象
                datetime_str = f"{date_str}_{time_str}"
                return datetime.strptime(datetime_str, "%Y%m%d_%H%M%S")
            else:
                # 如果无法提取日期，返回一个默认的最早时间
                return datetime.min
        except Exception as e:
            print(f"无法从路径 {file_path} 提取日期时间: {e}")
            return datetime.min
        
    def add_feature_extractor(self, feature_name: str, extractor_func: Callable):
        """
        添加特征提取函数
        
        Args:
            feature_name: 特征名称
            extractor_func: 特征提取函数，接收DataFrame，返回单个值或Series
        """
        self.feature_extractors[feature_name] = extractor_func
    
    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        """
        处理单个CSV文件
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            包含处理结果的字典
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            print(f"正在处理文件: {file_path}")
            print(f"文件形状: {df.shape}")
            
            result = {
                'file_path': file_path,
                'datetime': self.extract_datetime_from_path(file_path),
                'original_shape': df.shape,
                'columns': list(df.columns),
                'extracted_features': {}
            }
            
            # ==================== 特征提取区域 ====================
            
            # 应用自定义特征提取函数
            for feature_name, extractor in self.feature_extractors.items():
                try:
                    extracted = extractor(df)
                    if isinstance(extracted, dict):
                        result['extracted_features'].update(extracted)
                    else:
                        result['extracted_features'][feature_name] = extracted
                    print(f"  提取特征: {feature_name}")
                except Exception as e:
                    print(f"  提取特征 {feature_name} 时出错: {e}")
            
            # ============== 针对每个列的自定义特征提取区域 ==============
            
            # max_cell_voltage 特征
            if 'max_cell_voltage' in df.columns:
                coly = 'max_cell_voltage'
                colx = 'available_capacity'
                
                # 检查colx是否存在
                if colx in df.columns:
                    # 移除包含NaN的行
                    clean_data = df[[colx, coly]].dropna()
                    
                    if len(clean_data) > 1:  # 至少需要2个点进行线性回归
                        try:
                            # 准备数据
                            X = clean_data[colx].values.reshape(-1, 1)
                            y = clean_data[coly].values
                            
                            # 进行线性回归
                            lr_model = LinearRegression()
                            lr_model.fit(X, y)
                            
                            # 获取回归系数（斜率）
                            slope = lr_model.coef_[0]
                            result['extracted_features'][f'{coly}_vs_{colx}_slope'] = slope
                            
                            # 计算R²系数
                            y_pred = lr_model.predict(X)
                            r2 = r2_score(y, y_pred)
                            result['extracted_features'][f'{coly}_vs_{colx}_r2'] = r2
                            
                            print(f"  提取线性回归特征: 斜率={slope:.6f}, R²={r2:.6f}")
                            
                        except Exception as e:
                            print(f"  线性回归计算失败: {e}")
                            # 如果线性回归失败，设置默认值
                            result['extracted_features'][f'{coly}_vs_{colx}_slope'] = 0
                            result['extracted_features'][f'{coly}_vs_{colx}_r2'] = 0
                            
                    else:
                        print(f"  数据点不足，无法进行线性回归")
                        result['extracted_features'][f'{coly}_vs_{colx}_slope'] = 0
                        result['extracted_features'][f'{coly}_vs_{colx}_r2'] = 0
                        
                else:
                    print(f"  列 {colx} 不存在，跳过线性回归")
                    result['extracted_features'][f'{coly}_vs_{colx}_slope'] = 0
                    result['extracted_features'][f'{coly}_vs_{colx}_r2'] = 0
                result['extracted_features'][f'{coly}_mean'] = df[coly].mean()
            # charge_current 特征
            if 'charge_current' in df.columns:
                col = 'charge_current'
                result['extracted_features'][f'{col}_unique'] = len(df[col].unique().tolist())
                result['extracted_features'][f'{col}_mode'] = df[col].mode().iloc[0] if not df[col].mode().empty else None
                result['extracted_features'][f'{col}_jumps'] = (df[col].diff().abs() > 0.1).sum()  # 充电电流跳变次数
                
            # max_temperature 特征
            if 'max_temperature' in df.columns:
                col = 'max_temperature'
                result['extracted_features'][f'{col}_unique'] = len(df[col].unique().tolist())
                result['extracted_features'][f'{col}_mode'] = df[col].mode().iloc[0] if not df[col].mode().empty else None
                result['extracted_features'][f'{col}_jumps'] = (df[col].diff().abs() > 0.1).sum()  
            
            # charging_duration_seconds 特征
            if 'charging_duration_seconds' in df.columns:
                coly = 'charging_duration_seconds'
                colx = 'available_capacity'
                
                # 检查colx是否存在
                if colx in df.columns:
                    # 移除包含NaN的行
                    clean_data = df[[colx, coly]].dropna()
                    
                    if len(clean_data) > 1:  # 至少需要2个点进行线性回归
                        try:
                            # 准备数据
                            X = clean_data[colx].values.reshape(-1, 1)
                            y = clean_data[coly].values
                            
                            # 进行线性回归
                            lr_model = LinearRegression()
                            lr_model.fit(X, y)
                            
                            # 获取回归系数（斜率）
                            slope = lr_model.coef_[0]
                            result['extracted_features'][f'{coly}_vs_{colx}_slope'] = slope
                            
                            print(f"  提取线性回归特征: 斜率={slope:.6f}, R²={r2:.6f}")
                            
                        except Exception as e:
                            print(f"  线性回归计算失败: {e}")
                            # 如果线性回归失败，设置默认值
                            result['extracted_features'][f'{coly}_vs_{colx}_slope'] = 0
                            
                    else:
                        print(f"  数据点不足，无法进行线性回归")
                        result['extracted_features'][f'{coly}_vs_{colx}_slope'] = 0
                        
                else:
                    print(f"  列 {colx} 不存在，跳过线性回归")
                    result['extracted_features'][f'{coly}_vs_{colx}_slope'] = 0
                result['extracted_features'][f'{coly}_mean'] = df[coly].mean()

            # interpolated_dqdv 特征
            if 'interpolated_dqdv' in df.columns:
                col = 'interpolated_dqdv'
                colx = 'available_capacity'  # 用于计算面积和峰值对应容量
                result['extracted_features'][f'{col}_max'] = df[col].max()
                # 最大值对应的capacity
                max_dqdv_idx = df[col].idxmax()
                if colx in df.columns:
                    peak_capacity = df.loc[max_dqdv_idx, colx]
                    result['extracted_features'][f'{col}_peak_capacity'] = peak_capacity
                else:
                    result['extracted_features'][f'{col}_peak_capacity'] = None

                
                # 新增特征：曲线围成的面积
                if colx in df.columns:
                    # 移除包含NaN的行
                    clean_data = df[[colx, col]].dropna()
                    # 按照容量排序确保计算面积的正确性
                    clean_data = clean_data.sort_values(by=colx)
                    x_values = clean_data[colx].values
                    y_values = clean_data[col].values
                    
                    # 使用numpy的trapz函数计算曲线下面积 
                    area = np.trapz(y_values, x_values)
                    result['extracted_features'][f'{col}_area_under_curve'] = area
                    
                    # 计算正值区域面积（如果有负值）
                    positive_mask = y_values > 0
                    if positive_mask.any():
                        # 对正值区域重新排序和计算
                        x_pos = x_values[positive_mask]
                        y_pos = y_values[positive_mask]
                        # 按x值排序
                        sort_idx = np.argsort(x_pos)
                        positive_area = np.trapz(y_pos[sort_idx], x_pos[sort_idx])
                        result['extracted_features'][f'{col}_positive_area'] = positive_area
                    else:
                        result['extracted_features'][f'{col}_positive_area'] = 0
                    
                    print(f"  计算dQ/dV曲线面积: 总面积={area:.6f}")
        
                    max_dqdv_idx = df[col].idxmax()
                    peak_capacity = df.loc[max_dqdv_idx, colx]
                    result['extracted_features'][f'{col}_peak_capacity'] = peak_capacity
                    
                    # 寻找峰值，设置最小高度为最大值的10%
                    min_height = df[col].max() * 0.1
                    peaks, properties = find_peaks(df[col].values, height=min_height, distance=5)
            
            return result
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return None
                        
    def process_all_files_by_subfolder(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        按子文件夹处理所有CSV文件
        
        Returns:
            字典，键为子文件夹名，值为该文件夹下的处理结果列表
        """
        subfolder_results = {}
        
        if not os.path.exists(self.directory):
            print(f"目录不存在: {self.directory}")
            return subfolder_results
        
        # 遍历子文件夹
        for subfolder in os.listdir(self.directory):
            subfolder_path = os.path.join(self.directory, subfolder)
            
            if os.path.isdir(subfolder_path):
                print(f"\n=== 处理子文件夹: {subfolder} ===")
                subfolder_results[subfolder] = []
                
                # 遍历该子文件夹下的所有CSV文件
                for root, dirs, files in os.walk(subfolder_path):
                    for filename in files:
                        if filename.endswith('.csv'):
                            file_path = os.path.join(root, filename)
                            result = self.process_single_file(file_path)
                            if result is not None:
                                subfolder_results[subfolder].append(result)
                                self.processed_files.append(file_path)
                
                print(f"子文件夹 {subfolder} 处理了 {len(subfolder_results[subfolder])} 个文件")
        
        total_files = sum(len(results) for results in subfolder_results.values())
        print(f"\n总共处理了 {total_files} 个文件，分布在 {len(subfolder_results)} 个子文件夹中")
        return subfolder_results
    
    def save_features_by_subfolder(self, subfolder_results: Dict[str, List[Dict[str, Any]]], output_directory: str):
        """
        为每个子文件夹保存单独的特征CSV文件，按日期时间排序，最终输出不包含路径
        
        Args:
            subfolder_results: 按子文件夹分组的处理结果
            output_directory: 输出目录
        """
        # 创建输出目录
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        for subfolder_name, results in subfolder_results.items():
            if results:  # 如果有处理结果
                all_features = []
                
                # 按日期时间排序
                sorted_results = sorted(results, key=lambda x: x.get('datetime', datetime.min))
                
                for result in sorted_results:
                    if result and 'extracted_features' in result:
                        # 只保存extracted_features，不包含file_path
                        feature_row = {}
                        feature_row.update(result['extracted_features'])
                        all_features.append(feature_row)
                
                if all_features:
                    features_df = pd.DataFrame(all_features)
                    output_path = os.path.join(output_directory, f"{subfolder_name}_features.csv")
                    features_df.to_csv(output_path, index=False)
                    print(f"保存子文件夹 {subfolder_name} 的特征到: {output_path}")
                    print(f"  文件数量: {len(all_features)}, 特征数量: {len(features_df.columns)}")
                    print(f"  按时间排序: 从 {sorted_results[0]['datetime'].strftime('%Y-%m-%d %H:%M:%S')} 到 {sorted_results[-1]['datetime'].strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    # 设置目录路径
    input_directory = 'combined_features'
    output_directory = 'extracted_features_by_subfolder'
    
    # 创建数据处理器
    processor = DataProcessor(input_directory)
    
    # 按子文件夹处理所有文件
    subfolder_results = processor.process_all_files_by_subfolder()
    
    # 为每个子文件夹保存特征CSV
    if subfolder_results:
        processor.save_features_by_subfolder(subfolder_results, output_directory)
    
    print("特征提取完成！")

if __name__ == "__main__":
    main()