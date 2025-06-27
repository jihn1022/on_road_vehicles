import numpy as np
import pandas as pd
import os
import datetime as dt
from datetime import datetime
from natsort import ns, natsorted

def find_charging_sessions_in_file(file):
    """提取充电会话，返回充电数据片段"""
    cha = file
    cha.reset_index(drop=True, inplace=True)
    
    # 优化时间转换
    if not cha.empty and 'record_time' in cha.columns:
        cha_time = pd.to_datetime(cha['record_time'].astype(str), format='%Y%m%d%H%M%S')
    else:
        return []

    # 确保有足够的数据进行比较
    if len(cha_time) < 2:
        if not cha.empty:
            return [cha.copy()]
        return []

    # 按时间间隔分割片段（15秒间隔，以容纳10-12秒的正常间隔）
    interval = dt.timedelta(seconds=20)
    original_indices = np.where(cha_time.diff() > interval)[0]
    
    cha_list = []
    start_idx = 0
    for end_idx in original_indices:
        if start_idx < end_idx:
             cha_list.append(cha.iloc[start_idx:end_idx])
        start_idx = end_idx
    if start_idx < len(cha):
        cha_list.append(cha.iloc[start_idx:])
    
    if original_indices.size == 0 and not cha.empty:
        cha_list = [cha.copy()]

    # 过滤充电片段
    charging_sessions = []
    for cha_cut in cha_list:
        if cha_cut.empty:
            continue
            
        cha_cut = cha_cut.copy()
        cha_cut.reset_index(drop=True, inplace=True)
        
        # 数据量检查
        if len(cha_cut) < 100:
            continue
        
        # SOC变化检查
        soc_series = cha_cut['soc']
        if len(soc_series) < 2:
            continue
        dif_soc = soc_series.diff().iloc[1:]
        if dif_soc.empty:
            continue

        # 检查SOC跳变
        if not dif_soc.empty and (np.any(dif_soc > 2) or np.any(dif_soc < -0.1)):
            continue
            
        # 检查是否为充电状态（电流 < 0）
        current = cha_cut['charge_current']
        if not current.empty and np.any(current >= 0):
            continue
            
        # 检查max_cell_voltage列是否存在且有效
        if 'max_cell_voltage' not in cha_cut.columns:
            continue
            
        voltage_data = cha_cut['max_cell_voltage']
        if voltage_data.isnull().all():
            continue
            
        charging_sessions.append(cha_cut)
        
    return charging_sessions

def process_vehicle_file(file_path):
    """处理单个车辆文件"""
    try:
        file = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error reading CSV {file_path}: {e}")
        return []

    expected_columns = ['number','record_time','soc','pack_voltage','charge_current','max_cell_voltage',
                        'min_cell_voltage','max_temperature','min_temperature','available_energy','available_capacity']
    
    if list(file.columns) != expected_columns:
        if len(file.columns) == len(expected_columns):
            file.columns = expected_columns
        else:
            print(f"Error: Column count mismatch in {file_path}. Expected {len(expected_columns)}, got {len(file.columns)}")
            return []

    file = file.sort_values(by='record_time')
    file.reset_index(drop=True, inplace=True)
    
    charging_sessions = find_charging_sessions_in_file(file)
    return charging_sessions

def save_charging_voltage_curves(vehicle_id, charging_sessions, output_base_path):
    """保存每次充电的电压曲线数据"""
    vehicle_folder = os.path.join(output_base_path, vehicle_id)
    os.makedirs(vehicle_folder, exist_ok=True)
    
    saved_count = 0
    for session_idx, session_data in enumerate(charging_sessions):
        if session_data.empty:
            continue
            
        # 保留所有原始数据列
        voltage_curve_data = session_data.copy()
        
        # 确保时间列是datetime类型
        voltage_curve_data['record_time'] = pd.to_datetime(voltage_curve_data['record_time'].astype(str), format='%Y%m%d%H%M%S')
        
        # 添加计算字段
        voltage_curve_data['charging_duration_seconds'] = (
            voltage_curve_data['record_time'] - voltage_curve_data['record_time'].iloc[0]
        ).dt.total_seconds()
        
        # 生成文件名（包含开始时间）
        start_time = voltage_curve_data['record_time'].iloc[0]
        filename = f"charging_session_{session_idx+1:03d}_{start_time.strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(vehicle_folder, filename)
        
        try:
            voltage_curve_data.to_csv(filepath, index=False)
            saved_count += 1
            print(f"  Saved charging session {session_idx+1}: {filename}")
        except Exception as e:
            print(f"  Error saving charging session {session_idx+1}: {e}")
    
    return saved_count

if __name__ == '__main__':
    main_path = 'vehicles/'
    output_data_path = 'charging_voltage_curves/'
    
    # 创建输出目录
    os.makedirs(output_data_path, exist_ok=True)
    
    # 获取所有车辆文件
    vehicle_files = os.listdir(main_path)
    vehicle_files = natsorted(vehicle_files, alg=ns.PATH)
    
    total_sessions = 0
    total_vehicles_processed = 0
    
    print(f"开始处理 {len(vehicle_files)} 个车辆文件...")
    print(f"输出目录: {output_data_path}")
    
    for veh_idx, veh_filename_csv in enumerate(vehicle_files):
        if not veh_filename_csv.endswith('.csv'):
            continue
            
        vehicle_id = veh_filename_csv.replace('.csv', '')
        print(f"\n处理车辆 {veh_idx + 1}/{len(vehicle_files)}: {vehicle_id}")
        
        file_path = os.path.join(main_path, veh_filename_csv)
        charging_sessions = process_vehicle_file(file_path)
        
        if not charging_sessions:
            print(f"  车辆 {vehicle_id} 没有找到有效的充电数据")
            continue
        
        print(f"  找到 {len(charging_sessions)} 个充电会话")
        
        # 保存充电电压曲线
        saved_count = save_charging_voltage_curves(vehicle_id, charging_sessions, output_data_path)
        
        if saved_count > 0:
            total_sessions += saved_count
            total_vehicles_processed += 1
            print(f"  成功保存 {saved_count} 个充电会话的电压曲线")
        else:
            print(f"  车辆 {vehicle_id} 没有保存任何充电会话")
    
    print(f"\n处理完成!")
    print(f"处理的车辆数量: {total_vehicles_processed}")
    print(f"提取的充电会话总数: {total_sessions}")
    print(f"输出文件保存在: {output_data_path}")
    
    # 显示目录结构示例
    if total_vehicles_processed > 0:
        print(f"\n目录结构示例:")
        print(f"{output_data_path}/")
        sample_dirs = os.listdir(output_data_path)[:3]
        for sample_dir in sample_dirs:
            if os.path.isdir(os.path.join(output_data_path, sample_dir)):
                print(f"├── {sample_dir}/")
                sample_files = os.listdir(os.path.join(output_data_path, sample_dir))[:3]
                for i, sample_file in enumerate(sample_files):
                    if i == len(sample_files) - 1:
                        print(f"│   └── {sample_file}")
                    else:
                        print(f"│   ├── {sample_file}")
                if len(sample_files) > 3:
                    print(f"│   └── ... (还有 {len(sample_files)-3} 个文件)")