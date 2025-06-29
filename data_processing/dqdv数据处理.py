from scipy.signal import savgol_filter
import pandas as pd
import os
import numpy as np

def clean_data(df):
    # Keep only max_cell_voltage and available_capacity columns
    df = df[['max_cell_voltage', 'available_capacity']]
    # Remove rows with duplicate values in 'max_cell_voltage'
    df = df.drop_duplicates(subset=['max_cell_voltage'])
    # Calculate dq/dv and store result
    df['dq/dv'] = df['available_capacity'].diff() / df['max_cell_voltage'].diff()
    # Remove missing and negative values in dq/dv
    df = df[df['dq/dv'].notna() & (df['dq/dv'] >= 0)]
    # Apply Savitzky-Golay filter to smooth dq/dv values
    df['dq/dv'] = savgol_filter(df['dq/dv'], window_length=21, polyorder=2, mode='nearest')
    return df

def get_all_csv_files(root_folder):
    """递归获取所有子文件夹中的CSV文件"""
    csv_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def create_output_path(input_file, input_folder, output_folder):
    """根据输入文件路径创建对应的输出文件路径"""
    # 获取相对于输入文件夹的相对路径
    rel_path = os.path.relpath(input_file, input_folder)
    # 创建输出文件路径
    output_file = os.path.join(output_folder, rel_path)
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    return output_file

def process_single_file(csv_file):
    """处理单个CSV文件"""
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 检查必要的列是否存在
        required_columns = ['max_cell_voltage', 'available_capacity']
        if not all(col in df.columns for col in required_columns):
            print(f"警告: {csv_file} 缺少必要的列: {required_columns}")
            return None
        
        # 清理数据
        df_clean = clean_data(df.copy())
        
        # 检查清理后的数据是否为空
        if df_clean.empty:
            print(f"警告: {csv_file} 清理后数据为空")
            return None
            
        return df_clean
        
    except Exception as e:
        print(f"处理文件 {csv_file} 时出错: {str(e)}")
        return None

# 设置输入和输出文件夹路径
input_folder = 'charging_voltage_curves'
output_folder = 'processed_dqdv'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 递归获取所有CSV文件
csv_files = get_all_csv_files(input_folder)

print(f"找到 {len(csv_files)} 个CSV文件")
print("开始批量处理...")

# 统计处理结果
processed_count = 0
error_count = 0
empty_count = 0
failed_files = []  # 添加失败文件列表

# 遍历所有CSV文件进行处理
for i, csv_file in enumerate(csv_files, 1):
    print(f"\n[{i}/{len(csv_files)}] 处理文件: {os.path.relpath(csv_file, input_folder)}")
    
    # 处理单个文件
    df_processed = process_single_file(csv_file)
    
    if df_processed is not None:
        # 创建输出文件路径
        output_file = create_output_path(csv_file, input_folder, output_folder)
        
        # 保存处理后的数据
        df_processed.to_csv(output_file, index=False)
        
        processed_count += 1
        print(f"✓ 已保存到: {os.path.relpath(output_file, output_folder)}")
        print(f"  数据形状: {df_processed.shape}")
        print(f"  dq/dv 统计: 最小值={df_processed['dq/dv'].min():.6f}, "
              f"最大值={df_processed['dq/dv'].max():.6f}, "
              f"平均值={df_processed['dq/dv'].mean():.6f}")
        
    else:
        error_count += 1
        failed_files.append(os.path.relpath(csv_file, input_folder))  # 添加失败文件到列表
        print(f"✗ 处理失败")

print(f"\n" + "="*60)
print(f"批量处理完成!")
print(f"总文件数: {len(csv_files)}")
print(f"成功处理: {processed_count} 个文件")
print(f"处理失败: {error_count} 个文件")
print(f"成功率: {processed_count/len(csv_files)*100:.1f}%")

# 打印失败文件列表
if failed_files:
    print(f"\n失败文件列表:")
    for i, failed_file in enumerate(failed_files, 1):
        print(f"  {i}. {failed_file}")

print(f"输入文件夹: {input_folder}")
print(f"输出文件夹: {output_folder}")


# 如果有成功处理的文件，显示一个示例
if processed_count > 0:
    print(f"\n示例输出文件预览:")
    try:
        # 读取第一个成功处理的文件作为示例
        example_files = get_all_csv_files(output_folder)
        if example_files:
            example_df = pd.read_csv(example_files[0])
            print(f"文件: {os.path.basename(example_files[0])}")
            print(example_df.head())
    except:
        pass