import pandas as pd
import numpy as np
import os

def preprocess_interpolated_file(interpolated_file):
    """
    预处理单个插值文件，重命名列名
    """
    try:
        df = pd.read_csv(interpolated_file)
        
        # 重命名列名
        if 'capacity_integer' in df.columns and 'interpolated_dqdv' in df.columns:
            df = df.rename(columns={
                'capacity_integer': 'available_capacity',
                'interpolated_dqdv': 'dq/dv'
            })
        
        return df
        
    except Exception as e:
        print(f"预处理插值文件时出错: {str(e)}")
        return None

def merge_dqdv_data(original_file, interpolated_file):
    """
    将插值后的dq/dv数据合并到原始文件中
    直接将插值数据作为新列添加
    """
    try:
        # 读取原始文件
        original_df = pd.read_csv(original_file)
        
        # 预处理插值文件（重命名列名）
        interpolated_df = preprocess_interpolated_file(interpolated_file)
        
        if interpolated_df is None:
            print(f"警告: 无法预处理插值文件 {interpolated_file}")
            return None
        
        # 检查必要的列是否存在
        if 'available_capacity' not in original_df.columns:
            print(f"警告: {original_file} 缺少 available_capacity 列")
            return None
            
        if 'available_capacity' not in interpolated_df.columns or 'dq/dv' not in interpolated_df.columns:
            print(f"警告: {interpolated_file} 重命名后仍缺少必要的列")
            return None
        
        # 直接将插值数据的dq/dv列添加到原始数据中
        # 确保两个数据框的长度一致
        if len(original_df) == len(interpolated_df):
            original_df['interpolated_dqdv'] = interpolated_df['dq/dv'].values
            matched_count = len(original_df)
            total_count = len(original_df)
        else:
            print(f"警告: 数据长度不匹配 - 原始数据: {len(original_df)}, 插值数据: {len(interpolated_df)}")
            # 取较短的长度
            min_length = min(len(original_df), len(interpolated_df))
            original_df = original_df.iloc[:min_length].copy()
            original_df['interpolated_dqdv'] = interpolated_df['dq/dv'].iloc[:min_length].values
            matched_count = min_length
            total_count = min_length
        
        print(f"  匹配成功: {matched_count}/{total_count} 行")
        
        return original_df
        
    except Exception as e:
        print(f"合并文件时出错: {str(e)}")
        return None

def get_all_csv_files(folder):
    """获取文件夹中所有CSV文件，按路径排序"""
    csv_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return sorted(csv_files)

def create_output_path(input_file, input_folder, output_folder):
    """根据输入文件路径创建对应的输出文件路径"""
    # 获取相对于输入文件夹的相对路径
    rel_path = os.path.relpath(input_file, input_folder)
    # 在文件名前添加 "combined_" 前缀
    dir_name = os.path.dirname(rel_path)
    file_name = os.path.basename(rel_path)
    name, ext = os.path.splitext(file_name)
    new_file_name = f"combined_{name}{ext}"
    
    # 创建输出文件路径
    output_file = os.path.join(output_folder, dir_name, new_file_name)
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    return output_file

# 设置文件夹路径
input_folder = '/home/jihn/battery-charging-data-of-on-road-electric-vehicles/input_data'
interpolated_folder = '/home/jihn/battery-charging-data-of-on-road-electric-vehicles/interpolated_dqdv'
output_folder = '/home/jihn/battery-charging-data-of-on-road-electric-vehicles/combined_features'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 获取所有CSV文件
print("获取所有CSV文件...")
input_files = get_all_csv_files(input_folder)
interpolated_files = get_all_csv_files(interpolated_folder)

print(f"输入文件夹中找到 {len(input_files)} 个CSV文件")
print(f"插值文件夹中找到 {len(interpolated_files)} 个CSV文件")

# 检查文件数量是否匹配
if len(input_files) != len(interpolated_files):
    print(f"警告: 文件数量不匹配！输入文件: {len(input_files)}, 插值文件: {len(interpolated_files)}")
    min_count = min(len(input_files), len(interpolated_files))
    print(f"将处理前 {min_count} 个文件")
    input_files = input_files[:min_count]
    interpolated_files = interpolated_files[:min_count]

print("开始合并特征...")

# 统计处理结果
processed_count = 0
error_count = 0

# 按顺序一一对应处理文件
for i, (input_file, interpolated_file) in enumerate(zip(input_files, interpolated_files), 1):
    print(f"\n[{i}/{len(input_files)}] 处理文件对:")
    print(f"  输入文件: {os.path.relpath(input_file, input_folder)}")
    print(f"  插值文件: {os.path.relpath(interpolated_file, interpolated_folder)}")
    
    # 合并数据
    combined_df = merge_dqdv_data(input_file, interpolated_file)
    
    if combined_df is not None:
        # 创建输出文件路径
        output_file = create_output_path(input_file, input_folder, output_folder)
        
        # 保存合并后的数据
        combined_df.to_csv(output_file, index=False)
        
        processed_count += 1
        print(f"✓ 已保存到: {os.path.relpath(output_file, output_folder)}")
        print(f"  数据形状: {combined_df.shape}")
        print(f"  列名: {list(combined_df.columns)}")
        
        # 显示插值dq/dv的统计信息
        dqdv_values = combined_df['interpolated_dqdv'].dropna()
        if len(dqdv_values) > 0:
            print(f"  插值dq/dv统计: 最小值={dqdv_values.min():.6f}, "
                  f"最大值={dqdv_values.max():.6f}, "
                  f"平均值={dqdv_values.mean():.6f}")
        else:
            print("  警告: 没有成功匹配的dq/dv数据")
        
    else:
        error_count += 1
        print(f"✗ 合并失败")

print(f"\n" + "="*60)
print(f"特征合并完成!")
print(f"总文件数: {len(input_files)}")
print(f"成功处理: {processed_count} 个文件")
print(f"处理失败: {error_count} 个文件")
if len(input_files) > 0:
    print(f"成功率: {processed_count/len(input_files)*100:.1f}%")
print(f"输入文件夹: {input_folder}")
print(f"插值文件夹: {interpolated_folder}")
print(f"输出文件夹: {output_folder}")

# 如果有成功处理的文件，显示一个示例
if processed_count > 0:
    print(f"\n示例合并结果预览:")
    try:
        # 找到第一个输出文件作为示例
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                if file.endswith('.csv'):
                    example_file = os.path.join(root, file)
                    example_df = pd.read_csv(example_file)
                    print(f"文件: {os.path.basename(example_file)}")
                    print(f"列名: {list(example_df.columns)}")
                    print(example_df.head(10))
                    
                    # 显示插值列的匹配情况
                    total_rows = len(example_df)
                    matched_rows = example_df['interpolated_dqdv'].notna().sum()
                    print(f"匹配情况: {matched_rows}/{total_rows} 行有插值数据")
                    break
            break
    except Exception as e:
        print(f"显示示例时出错: {str(e)}")