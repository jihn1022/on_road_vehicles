import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os

def interpolate_dqdv(df):
    """
    根据capacity和dq/dv进行插值
    capacity作为x值，在其范围内的所有整数点进行插值
    """
    # 提取数据
    capacity = df['available_capacity'].values
    dqdv = df['dq/dv'].values
    
    # 移除NaN值
    mask = ~(np.isnan(capacity) | np.isnan(dqdv))
    capacity_clean = capacity[mask]
    dqdv_clean = dqdv[mask]
    
    if len(capacity_clean) < 2:
        print("警告: 有效数据点不足，无法进行插值")
        return None
    
    # 确定插值范围（capacity的最小值到最大值之间的所有整数）
    min_capacity = int(np.ceil(capacity_clean.min()))
    max_capacity = int(np.floor(capacity_clean.max()))
    
    if min_capacity >= max_capacity:
        print(f"警告: 插值范围无效 (min: {min_capacity}, max: {max_capacity})")
        return None
    
    # 生成插值点（整数序列）
    interpolation_points = np.arange(min_capacity, max_capacity + 1)
    
    # 创建插值函数（使用线性插值）
    try:
        # 如果有重复的capacity值，先去重
        if len(capacity_clean) != len(np.unique(capacity_clean)):
            # 按capacity排序并去重
            sorted_indices = np.argsort(capacity_clean)
            capacity_clean = capacity_clean[sorted_indices]
            dqdv_clean = dqdv_clean[sorted_indices]
            
            # 去除重复的capacity值，保留第一个
            unique_capacity = []
            unique_dqdv = []
            for i, cap in enumerate(capacity_clean):
                if i == 0 or cap != capacity_clean[i-1]:
                    unique_capacity.append(cap)
                    unique_dqdv.append(dqdv_clean[i])
            
            capacity_clean = np.array(unique_capacity)
            dqdv_clean = np.array(unique_dqdv)
        
        interp_func = interp1d(capacity_clean, dqdv_clean, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
        
        # 进行插值
        interpolated_dqdv = interp_func(interpolation_points)
        
        # 创建插值结果DataFrame
        result_df = pd.DataFrame({
            'capacity_integer': interpolation_points,
            'interpolated_dqdv': interpolated_dqdv
        })
        
        return result_df
        
    except Exception as e:
        print(f"插值过程中出错: {str(e)}")
        return None

def detect_anomalies(df, file_path):
    """
    检测插值后数据的异常值
    返回异常信息字典
    """
    anomalies = {
        'file_path': file_path,
        'negative_values': [],
        'extreme_values': [],
        'nan_values': [],
        'statistics': {}
    }
    
    dqdv_values = df['interpolated_dqdv'].values
    capacity_values = df['capacity_integer'].values
    
    # 检测负值
    negative_mask = dqdv_values < 0
    if np.any(negative_mask):
        negative_positions = np.where(negative_mask)[0]
        for pos in negative_positions:
            anomalies['negative_values'].append({
                'position': int(pos),
                'capacity': int(capacity_values[pos]),
                'dqdv_value': float(dqdv_values[pos])
            })
    
    # 检测NaN值
    nan_mask = np.isnan(dqdv_values)
    if np.any(nan_mask):
        nan_positions = np.where(nan_mask)[0]
        for pos in nan_positions:
            anomalies['nan_values'].append({
                'position': int(pos),
                'capacity': int(capacity_values[pos])
            })
    
    # 检测极端值（超过3倍标准差）
    valid_values = dqdv_values[~(negative_mask | nan_mask)]
    if len(valid_values) > 0:
        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values)
        threshold = mean_val + 3 * std_val
        
        extreme_mask = dqdv_values > threshold
        if np.any(extreme_mask):
            extreme_positions = np.where(extreme_mask)[0]
            for pos in extreme_positions:
                anomalies['extreme_values'].append({
                    'position': int(pos),
                    'capacity': int(capacity_values[pos]),
                    'dqdv_value': float(dqdv_values[pos]),
                    'threshold': float(threshold)
                })
    
    # 统计信息
    anomalies['statistics'] = {
        'total_points': len(dqdv_values),
        'negative_count': len(anomalies['negative_values']),
        'nan_count': len(anomalies['nan_values']),
        'extreme_count': len(anomalies['extreme_values']),
        'min_value': float(np.nanmin(dqdv_values)),
        'max_value': float(np.nanmax(dqdv_values)),
        'mean_value': float(np.nanmean(dqdv_values)),
        'std_value': float(np.nanstd(dqdv_values))
    }
    
    return anomalies

def process_single_file(csv_file):
    """
    处理单个CSV文件
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 检查必要的列是否存在
        required_columns = ['available_capacity', 'dq/dv']
        if not all(col in df.columns for col in required_columns):
            print(f"警告: {csv_file} 缺少必要的列: {required_columns}")
            return None, None
        
        # 进行插值
        interpolated_df = interpolate_dqdv(df)
        
        if interpolated_df is None:
            print(f"警告: {csv_file} 插值失败")
            return None, None
        
        # 检测异常值
        anomalies = detect_anomalies(interpolated_df, csv_file)
        
        return interpolated_df, anomalies
        
    except Exception as e:
        print(f"处理文件 {csv_file} 时出错: {str(e)}")
        return None, None

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
    # 在文件名前添加 "interpolated_" 前缀
    dir_name = os.path.dirname(rel_path)
    file_name = os.path.basename(rel_path)
    name, ext = os.path.splitext(file_name)
    new_file_name = f"interpolated_{name}{ext}"
    
    # 创建输出文件路径
    output_file = os.path.join(output_folder, dir_name, new_file_name)
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    return output_file

def save_anomaly_report(all_anomalies, output_folder):
    """保存异常检测报告"""
    report_file = os.path.join(output_folder, 'anomaly_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("dq/dv 插值异常检测报告\n")
        f.write("=" * 60 + "\n\n")
        
        # 总体统计
        total_files = len(all_anomalies)
        files_with_negatives = sum(1 for a in all_anomalies if a['statistics']['negative_count'] > 0)
        files_with_extremes = sum(1 for a in all_anomalies if a['statistics']['extreme_count'] > 0)
        files_with_nans = sum(1 for a in all_anomalies if a['statistics']['nan_count'] > 0)
        
        f.write(f"总体统计:\n")
        f.write(f"  处理文件总数: {total_files}\n")
        f.write(f"  包含负值的文件: {files_with_negatives}\n")
        f.write(f"  包含极端值的文件: {files_with_extremes}\n")
        f.write(f"  包含NaN值的文件: {files_with_nans}\n\n")
        
        # 详细异常信息
        for anomaly in all_anomalies:
            if (anomaly['statistics']['negative_count'] > 0 or 
                anomaly['statistics']['extreme_count'] > 0 or 
                anomaly['statistics']['nan_count'] > 0):
                
                f.write(f"文件: {os.path.basename(anomaly['file_path'])}\n")
                f.write(f"路径: {anomaly['file_path']}\n")
                f.write(f"统计: 总点数={anomaly['statistics']['total_points']}, "
                       f"负值={anomaly['statistics']['negative_count']}, "
                       f"极端值={anomaly['statistics']['extreme_count']}, "
                       f"NaN值={anomaly['statistics']['nan_count']}\n")
                
                # 负值详情
                if anomaly['negative_values']:
                    f.write("  负值位置:\n")
                    for neg in anomaly['negative_values'][:10]:  # 只显示前10个
                        f.write(f"    容量={neg['capacity']}, dq/dv={neg['dqdv_value']:.6f}\n")
                    if len(anomaly['negative_values']) > 10:
                        f.write(f"    ... 还有 {len(anomaly['negative_values']) - 10} 个负值\n")
                
                # 极端值详情
                if anomaly['extreme_values']:
                    f.write("  极端值位置:\n")
                    for ext in anomaly['extreme_values'][:5]:  # 只显示前5个
                        f.write(f"    容量={ext['capacity']}, dq/dv={ext['dqdv_value']:.6f} "
                               f"(阈值={ext['threshold']:.6f})\n")
                    if len(anomaly['extreme_values']) > 5:
                        f.write(f"    ... 还有 {len(anomaly['extreme_values']) - 5} 个极端值\n")
                
                f.write("\n")
    
    return report_file

def save_failed_files_report(failed_files, output_folder):
    """保存插值失败文件报告"""
    if not failed_files:
        return None
        
    report_file = os.path.join(output_folder, 'interpolation_failed_files.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("插值失败文件列表\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"失败文件总数: {len(failed_files)}\n\n")
        
        for i, failed_info in enumerate(failed_files, 1):
            f.write(f"{i}. 文件名: {failed_info['filename']}\n")
            f.write(f"   完整路径: {failed_info['filepath']}\n")
            f.write(f"   失败原因: {failed_info['reason']}\n")
            if 'error_details' in failed_info:
                f.write(f"   错误详情: {failed_info['error_details']}\n")
            f.write("\n")
    
    return report_file

# 设置输入和输出文件夹路径
input_folder = '/home/jihn/battery-charging-data-of-on-road-electric-vehicles/processed_dqdv'
output_folder = '/home/jihn/battery-charging-data-of-on-road-electric-vehicles/interpolated_dqdv'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 递归获取所有CSV文件
csv_files = get_all_csv_files(input_folder)

print(f"找到 {len(csv_files)} 个CSV文件")
print("开始批量插值处理...")

# 统计处理结果
processed_count = 0
error_count = 0
all_anomalies = []
failed_files = []  # 新增：收集失败文件信息

# 遍历所有CSV文件进行处理
for i, csv_file in enumerate(csv_files, 1):
    print(f"\n[{i}/{len(csv_files)}] 处理文件: {os.path.relpath(csv_file, input_folder)}")
    
    # 处理单个文件
    df_interpolated, anomalies = process_single_file(csv_file)
    
    if df_interpolated is not None:
        # 创建输出文件路径
        output_file = create_output_path(csv_file, input_folder, output_folder)
        
        # 保存插值后的数据
        df_interpolated.to_csv(output_file, index=False)
        
        # 保存异常信息
        all_anomalies.append(anomalies)
        
        processed_count += 1
        print(f"✓ 已保存到: {os.path.relpath(output_file, output_folder)}")
        print(f"  插值数据形状: {df_interpolated.shape}")
        print(f"  容量范围: {df_interpolated['capacity_integer'].min()} - {df_interpolated['capacity_integer'].max()}")
        print(f"  dq/dv 统计: 最小值={anomalies['statistics']['min_value']:.6f}, "
              f"最大值={anomalies['statistics']['max_value']:.6f}, "
              f"平均值={anomalies['statistics']['mean_value']:.6f}")
        
        # 显示异常信息
        if anomalies['statistics']['negative_count'] > 0:
            print(f"  ⚠️  发现 {anomalies['statistics']['negative_count']} 个负值")
        if anomalies['statistics']['extreme_count'] > 0:
            print(f"  ⚠️  发现 {anomalies['statistics']['extreme_count']} 个极端值")
        if anomalies['statistics']['nan_count'] > 0:
            print(f"  ⚠️  发现 {anomalies['statistics']['nan_count']} 个NaN值")
        
    else:
        error_count += 1
        print(f"✗ 处理失败")
        
        # 新增：收集失败文件信息
        failed_info = {
            'filename': os.path.basename(csv_file),
            'filepath': csv_file,
            'relative_path': os.path.relpath(csv_file, input_folder),
            'reason': '插值失败或数据处理异常'
        }
        
        # 尝试获取更详细的失败原因
        try:
            df = pd.read_csv(csv_file)
            required_columns = ['available_capacity', 'dq/dv']
            if not all(col in df.columns for col in required_columns):
                failed_info['reason'] = f"缺少必要的列: {[col for col in required_columns if col not in df.columns]}"
            else:
                # 检查数据质量
                capacity = df['available_capacity'].values
                dqdv = df['dq/dv'].values
                mask = ~(np.isnan(capacity) | np.isnan(dqdv))
                valid_points = mask.sum()
                
                if valid_points < 2:
                    failed_info['reason'] = f"有效数据点不足 ({valid_points} < 2)"
                else:
                    capacity_clean = capacity[mask]
                    min_cap = int(np.ceil(capacity_clean.min()))
                    max_cap = int(np.floor(capacity_clean.max()))
                    if min_cap >= max_cap:
                        failed_info['reason'] = f"插值范围无效 (min: {min_cap}, max: {max_cap})"
                    else:
                        failed_info['reason'] = "插值过程中发生未知错误"
        except Exception as e:
            failed_info['reason'] = f"读取文件时出错: {str(e)}"
            failed_info['error_details'] = str(e)
        
        failed_files.append(failed_info)

# 保存异常检测报告
if all_anomalies:
    report_file = save_anomaly_report(all_anomalies, output_folder)
    print(f"\n异常检测报告已保存到: {report_file}")

# 新增：保存失败文件报告
if failed_files:
    failed_report_file = save_failed_files_report(failed_files, output_folder)
    print(f"插值失败文件报告已保存到: {failed_report_file}")

print(f"\n" + "="*60)
print(f"批量插值处理完成!")
print(f"总文件数: {len(csv_files)}")
print(f"成功处理: {processed_count} 个文件")
print(f"处理失败: {error_count} 个文件")
print(f"成功率: {processed_count/len(csv_files)*100:.1f}%")

# 新增：打印所有插值失败的文件名
if failed_files:
    print(f"\n" + "="*60)
    print(f"插值失败的文件列表 (共 {len(failed_files)} 个):")
    print("="*60)
    for i, failed_info in enumerate(failed_files, 1):
        print(f"{i:3d}. {failed_info['filename']}")
        print(f"     路径: {failed_info['relative_path']}")
        print(f"     原因: {failed_info['reason']}")
        print()
    
    # 按失败原因分类统计
    reason_stats = {}
    for failed_info in failed_files:
        reason = failed_info['reason']
        if '缺少必要的列' in reason:
            reason_key = '缺少必要列'
        elif '有效数据点不足' in reason:
            reason_key = '数据点不足'
        elif '插值范围无效' in reason:
            reason_key = '插值范围无效'
        elif '读取文件时出错' in reason:
            reason_key = '文件读取错误'
        else:
            reason_key = '其他错误'
        
        reason_stats[reason_key] = reason_stats.get(reason_key, 0) + 1
    
    print("失败原因统计:")
    for reason, count in reason_stats.items():
        print(f"  {reason}: {count} 个文件")

# 异常统计汇总
if all_anomalies:
    negative_count = sum(a['statistics']['negative_count'] for a in all_anomalies)
    extreme_count = sum(a['statistics']['extreme_count'] for a in all_anomalies)
    nan_count = sum(a['statistics']['nan_count'] for a in all_anomalies)
    files_with_negatives = sum(1 for a in all_anomalies if a['statistics']['negative_count'] > 0)
    
    print(f"\n异常值统计汇总:")
    print(f"总负值数量: {negative_count}")
    print(f"总极端值数量: {extreme_count}")
    print(f"总NaN值数量: {nan_count}")
    print(f"包含负值的文件数: {files_with_negatives}")

print(f"\n输入文件夹: {input_folder}")
print(f"输出文件夹: {output_folder}")

# 如果有成功处理的文件，显示一个示例
if processed_count > 0:
    print(f"\n示例插值结果预览:")
    try:
        # 读取第一个成功处理的文件作为示例
        example_files = get_all_csv_files(output_folder)
        if example_files:
            example_df = pd.read_csv(example_files[0])
            print(f"文件: {os.path.basename(example_files[0])}")
            print(example_df.head(10))
            print("...")
            print(example_df.tail(5))
    except:
        pass