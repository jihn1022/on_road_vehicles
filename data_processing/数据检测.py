import os
import pandas as pd
from datetime import datetime, timedelta
import re
from pathlib import Path

def parse_filename(filename):
    """解析文件名获取时间信息"""
    pattern = r'charging_session_(\d+)_(\d{8})_(\d{6})\.csv'
    match = re.match(pattern, filename)
    if match:
        session_id = match.group(1)
        date_str = match.group(2)  # YYYYMMDD
        time_str = match.group(3)  # HHMMSS
        
        # 解析日期时间
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        
        dt = datetime(year, month, day, hour, minute, second)
        return session_id, dt
    return None, None

def get_soc_range(csv_file_path):
    """获取CSV文件中SOC的变化范围"""
    try:
        df = pd.read_csv(csv_file_path)
        if 'soc' in df.columns:
            soc_min = df['soc'].min()
            soc_max = df['soc'].max()
            soc_range = soc_max - soc_min
            return soc_range, soc_min, soc_max
        else:
            print(f"警告: {csv_file_path} 中没有找到 'soc' 列")
            return None, None, None
    except Exception as e:
        print(f"读取文件 {csv_file_path} 时出错: {e}")
        return None, None, None

def find_csv_files(root_folder):
    """查找所有符合格式的CSV文件"""
    csv_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.csv') and 'charging_session_' in file:
                session_id, dt = parse_filename(file)
                if session_id and dt:
                    csv_files.append({
                        'path': os.path.join(root, file),
                        'filename': file,
                        'session_id': session_id,
                        'datetime': dt
                    })
    return csv_files

def group_files_by_time(csv_files):
    """按时间分组文件（同一天或4小时内）"""
    groups = []
    processed = set()
    
    for i, file1 in enumerate(csv_files):
        if i in processed:
            continue
            
        group = [file1]
        processed.add(i)
        
        for j, file2 in enumerate(csv_files):
            if j in processed or j <= i:
                continue
                
            dt1 = file1['datetime']
            dt2 = file2['datetime']
            
            # 检查是否同一天
            same_day = dt1.date() == dt2.date()
            
            # 检查是否在4小时内
            time_diff = abs((dt1 - dt2).total_seconds()) / 3600  # 转换为小时
            within_4_hours = time_diff <= 4
            
            if same_day or within_4_hours:
                group.append(file2)
                processed.add(j)
        
        if len(group) > 1:  # 只关心有多个文件的组
            groups.append(group)
    
    return groups

def detect_low_soc_change(root_folder):
    """主检测函数"""
    print(f"开始检测文件夹: {root_folder}")
    print("=" * 80)
    
    # 查找所有CSV文件
    csv_files = find_csv_files(root_folder)
    print(f"找到 {len(csv_files)} 个充电会话文件")
    
    if not csv_files:
        print("没有找到符合格式的CSV文件")
        return
    
    # 按时间分组
    groups = group_files_by_time(csv_files)
    print(f"找到 {len(groups)} 个时间相关的文件组")
    print("=" * 80)
    
    low_soc_files = []
    
    # 检查每个组中的文件
    for group_idx, group in enumerate(groups):
        print(f"\n组 {group_idx + 1}: {len(group)} 个文件")
        print("-" * 40)
        
        for file_info in group:
            soc_range, soc_min, soc_max = get_soc_range(file_info['path'])
            
            if soc_range is not None:
                print(f"文件: {file_info['filename']}")
                print(f"  时间: {file_info['datetime']}")
                print(f"  SOC范围: {soc_min:.1f} - {soc_max:.1f} (变化幅度: {soc_range:.1f})")
                
                if soc_range <= 1.5:
                    print(f"  ✓ SOC变化幅度 ({soc_range:.1f}) ≤ 1.5，记录此文件")
                    low_soc_files.append({
                        'file': file_info,
                        'soc_range': soc_range,
                        'soc_min': soc_min,
                        'soc_max': soc_max
                    })
                else:
                    print(f"  ✗ SOC变化幅度 ({soc_range:.1f}) > 3")     
                print()
    
    # 打印汇总结果
    print("=" * 80)
    print("检测结果汇总:")
    print(f"总共检测文件数: {len(csv_files)}")
    print(f"SOC变化幅度 ≤ 3 的文件数: {len(low_soc_files)}")
    
    if low_soc_files:
        print("\n符合条件的文件列表:")
        print("-" * 80)
        for idx, item in enumerate(low_soc_files, 1):
            file_info = item['file']
            print(f"{idx}. {file_info['filename']}")
            print(f"   路径: {file_info['path']}")
            print(f"   时间: {file_info['datetime']}")
            print(f"   SOC: {item['soc_min']:.1f} - {item['soc_max']:.1f} (变化: {item['soc_range']:.1f})")
            print()

# 使用示例
if __name__ == "__main__":
    # 设置您的文件夹路径
    root_folder = "charging_voltage_curves"
    
    # 执行检测
    detect_low_soc_change(root_folder)