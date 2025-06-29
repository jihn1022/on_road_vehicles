import os
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler
import warnings
import pickle
warnings.filterwarnings('ignore')

def extract_and_merge_csv(directory):
    """
    遍历文件夹的csv文件，读取每个csv文件，提取所有数据，并把所有相同列名的数据合并到一个DataFrame中
    
    Args:
        directory: 包含CSV文件的目录路径
        
    Returns:
        合并后的DataFrame
    """
    all_files = glob.glob(os.path.join(directory, "*.csv"))
    
    if not all_files:
        print(f"在目录 {directory} 中没有找到CSV文件")
        return pd.DataFrame()
    
    print(f"找到 {len(all_files)} 个CSV文件")
    
    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            # 添加源文件信息列
            df['source_file'] = os.path.basename(file).replace('.csv', '')
            print(f"读取文件: {os.path.basename(file)}, 形状: {df.shape}")
            df_list.append(df)
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")
    
    if df_list:
        merged_df = pd.concat(df_list, ignore_index=True)
        print(f"合并后的DataFrame形状: {merged_df.shape}")
        return merged_df
    else:
        return pd.DataFrame()

def normalize_columns(df):
    """
    对DataFrame中的每个数值列进行统一的min-max归一化
    
    Args:
        df: 输入的DataFrame（包含所有文件的合并数据）
        
    Returns:
        normalized_df: 归一化后的DataFrame
        scalers: 字典，包含每列的MinMaxScaler对象（用于后续逆变换）
    """
    normalized_df = df.copy()
    scalers = {}
    
    # 识别数值列（排除source_file列）
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'source_file' in numeric_columns:
        numeric_columns.remove('source_file')
    
    print(f"识别到 {len(numeric_columns)} 个数值列需要归一化")
    
    for column in numeric_columns:
        try:
            # 移除NaN值进行归一化
            non_nan_mask = ~df[column].isna()
            
            if non_nan_mask.sum() > 1:  # 至少需要2个非NaN值
                # 创建MinMaxScaler
                scaler = MinMaxScaler()
                
                # 对所有文件的该列非NaN值进行统一归一化
                non_nan_values = df.loc[non_nan_mask, column].values.reshape(-1, 1)
                normalized_values = scaler.fit_transform(non_nan_values)
                
                # 将归一化后的值放回原位置
                normalized_df.loc[non_nan_mask, column] = normalized_values.flatten()
                
                # 保存scaler以备后用
                scalers[column] = scaler
                
                print(f"列 '{column}' 归一化完成: 原始数据量={non_nan_mask.sum()}, 最小值={normalized_df[column].min():.6f}, 最大值={normalized_df[column].max():.6f}")
            else:
                print(f"列 '{column}' 的有效值不足，跳过归一化")
                
        except Exception as e:
            print(f"归一化列 '{column}' 时出错: {e}")
    
    return normalized_df, scalers

def save_normalized_data_by_source(normalized_df, scalers, output_directory):
    """
    将归一化后的数据按原始文件分别保存，并保存scaler信息
    
    Args:
        normalized_df: 归一化后的DataFrame（包含source_file列）
        scalers: scaler字典
        output_directory: 输出目录
    """
    # 创建输出目录
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # 保存统一的scaler信息
    scaler_file = os.path.join(output_directory, "scalers.pkl")
    with open(scaler_file, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"归一化器保存到: {scaler_file}")
    
    # 按source_file分组保存
    if 'source_file' in normalized_df.columns:
        for source_file in normalized_df['source_file'].unique():
            file_data = normalized_df[normalized_df['source_file'] == source_file].copy()
            # 删除source_file列
            file_data = file_data.drop('source_file', axis=1)
            
            # 保存文件
            output_file = os.path.join(output_directory, f"{source_file}_features.csv")
            file_data.to_csv(output_file, index=False)
            print(f"文件 {source_file} 的归一化数据保存到: {output_file}")
    else:
        # 如果没有source_file列，保存整个数据
        output_file = os.path.join(output_directory, "normalized_features.csv")
        normalized_df.to_csv(output_file, index=False)
        print(f"归一化数据保存到: {output_file}")

def main():
    """
    主函数：合并所有CSV文件后统一进行归一化处理
    """
    # 设置路径
    input_directory = 'extracted_features_by_subfolder'
    output_directory = 'normalized_features'
    
    print("=== 开始统一归一化处理 ===")
    
    if not os.path.exists(input_directory):
        print(f"输入目录不存在: {input_directory}")
        return
    
    # 创建主输出目录
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # 步骤1：合并所有CSV文件
    print("\n步骤1：合并所有CSV文件...")
    merged_df = extract_and_merge_csv(input_directory)
    
    if merged_df.empty:
        print("没有数据可处理")
        return
    
    # 步骤2：对合并后的数据进行统一归一化
    print("\n步骤2：对合并数据进行统一归一化...")
    normalized_df, scalers = normalize_columns(merged_df)
    
    # 步骤3：按原始文件分别保存归一化后的数据
    print("\n步骤3：按原始文件保存归一化数据...")
    save_normalized_data_by_source(normalized_df, scalers, output_directory)
    
    print("\n=== 统一归一化处理完成 ===")
    print(f"结果保存在: {output_directory}")
    print(f"共处理了 {len(scalers)} 个数值列")
    print("所有文件使用相同的归一化参数，确保数据一致性")

if __name__ == "__main__":
    main()