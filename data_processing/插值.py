import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# --- 核心插值函数 (与之前版本相同，已包含处理重复X值和排序的逻辑) ---
def interpolate_variable_segments_decimal(segments_data, interpolation_step_size=1.0, kind='cubic'):
    """
    对多段数据进行插值，每段的插值点数根据其x轴长度和指定的步长动态决定。
    支持原始数据和插值结果包含小数点，并处理同一X值对应多个Y值的情况（取平均）。

    参数:
    segments_data (list of dict): 包含多段数据的列表。
                                  每段数据是一个字典，包含 'x' 和 'y' 键，
                                  对应其原始x和y坐标的列表或numpy数组。
                                  示例: [{'x': [x1, x2, ...], 'y': [y1, y2, ...], 'segment_name': 'seg1'}, ...]
                                  'segment_name' 用于输出标识，这里将是CSV文件的相对路径。
    interpolation_step_size (float): 插值后x轴上的步长。例如，如果设置为0.1，
                                     则插值结果的x坐标将以0.1为间隔。
                                     如果设置为1.0，则以1.0为间隔。
                                     请确保此值大于0。
    kind (str): 插值方法。可以是 'linear' (线性插值), 'quadratic' (二次样条插值),
                'cubic' (三次样条插值) 等。对于平滑曲线，'cubic' 通常是更好的选择。
                注意：'cubic' 需要至少4个数据点，'quadratic' 需要至少3个。
                如果数据点不足，函数会自动回退到 'linear'。

    返回:
    list of dict: 包含每段插值结果的列表。
                  每段结果是一个字典，包含 'x_interpolated', 'y_interpolated' 键，
                  以及 'segment_name' (CSV文件的相对路径)。
    """
    
    if not isinstance(interpolation_step_size, (int, float)) or interpolation_step_size <= 0:
        raise ValueError("interpolation_step_size 必须是大于0的数字。")

    all_interpolated_segments = []

    for i, segment in enumerate(segments_data):
        # segment_name 现在是CSV文件的相对路径，例如 'subfolder/data.csv'
        segment_name = segment.get('segment_name', f"Unnamed_Segment_{i+1}") 
        original_x = np.array(segment['x'], dtype=float)
        original_y = np.array(segment['y'], dtype=float)

        # --- 处理重复的x值和排序 (使用Pandas更简洁和健壮) ---
        if len(original_x) == 0:
            print(f"警告: 文件 '{segment_name}' 原始数据为空。跳过此文件。")
            all_interpolated_segments.append({
                'x_interpolated': np.array([]),
                'y_interpolated': np.array([]),
                'segment_name': segment_name
            })
            continue

        df_segment = pd.DataFrame({'x': original_x, 'y': original_y})
        processed_data = df_segment.groupby('x')['y'].mean().reset_index().sort_values(by='x')

        original_x_processed = processed_data['x'].values
        original_y_processed = processed_data['y'].values
        # --- 结束处理重复的x值和排序 ---

        # 处理特殊情况：如果处理后的数据点少于2个，无法进行插值
        if len(original_x_processed) < 2:
            print(f"警告: 文件 '{segment_name}' 处理后的数据点 ({len(original_x_processed)}个) 不足以进行插值。将返回处理后的原始点。")
            interpolated_x = original_x_processed
            interpolated_y = original_y_processed
            all_interpolated_segments.append({
                'x_interpolated': interpolated_x,
                'y_interpolated': interpolated_y,
                'segment_name': segment_name
            })
            continue
        
        # 1. 确定每段数据的x轴范围
        x_min = original_x_processed.min()
        x_max = original_x_processed.max()

        # 2. 根据x轴范围内的整数点生成插值点
        x_min_int = int(np.ceil(x_min))  # 向上取整到最小整数
        x_max_int = int(np.floor(x_max))  # 向下取整到最大整数
        
        # 检查是否有有效的整数范围
        if x_min_int > x_max_int:
            print(f"警告: 文件 '{segment_name}' 的x轴范围 ({x_min:.3f} 到 {x_max:.3f}) 内没有整数点。将使用原数据点。")
            interpolated_x = original_x_processed
            interpolated_y = original_y_processed
            all_interpolated_segments.append({
                'x_interpolated': interpolated_x,
                'y_interpolated': interpolated_y,
                'segment_name': segment_name
            })
            continue
        
        # 生成x轴范围内的所有整数点
        interpolated_x = np.arange(x_min_int, x_max_int + 1, dtype=float)
        
        # 如果只有一个整数点，无法进行插值
        if len(interpolated_x) < 2:
            print(f"警告: 文件 '{segment_name}' 在x轴范围内只有 {len(interpolated_x)} 个整数点，无法进行插值。将返回处理后的原始点。")
            interpolated_x = original_x_processed
            interpolated_y = original_y_processed
            all_interpolated_segments.append({
                'x_interpolated': interpolated_x,
                'y_interpolated': interpolated_y,
                'segment_name': segment_name
            })
            continue

        # 3. 执行插值
        min_points_for_kind = 2 
        if kind == 'quadratic':
            min_points_for_kind = 3
        elif kind == 'cubic':
            min_points_for_kind = 4

        current_kind = kind
        if len(original_x_processed) < min_points_for_kind:
            print(f"警告: 文件 '{segment_name}' 数据点 ({len(original_x_processed)}个) 不足以进行 '{kind}' 插值。将回退到 'linear'。")
            current_kind = 'linear'
        
        f_interp = interp1d(original_x_processed, original_y_processed, kind=current_kind, fill_value="extrapolate")
        interpolated_y = f_interp(interpolated_x)
        
        print(f"文件 '{segment_name}': x范围 {x_min:.3f}-{x_max:.3f}, 插值到 {len(interpolated_x)} 个整数点 ({x_min_int}-{x_max_int})")
        
        all_interpolated_segments.append({
            'x_interpolated': interpolated_x,
            'y_interpolated': interpolated_y,
            'segment_name': segment_name # 存储CSV文件的相对路径
        })
        
    return all_interpolated_segments

# --- 文件系统遍历和数据读取函数 (修改为按CSV文件处理) ---
def process_input_folder_for_interpolation(input_root_folder, x_col_name, y_col_name, interpolation_step_size=1.0, kind='cubic'):
    """
    遍历指定输入根文件夹下的所有CSV文件，读取指定列数据，
    处理一X多Y的情况（取平均），然后对每个CSV文件的数据进行插值。

    参数:
    input_root_folder (str): 包含CSV文件（可能在子文件夹中）的根目录路径。
    x_col_name (str): CSV文件中作为X轴数据的列名。
    y_col_name (str): CSV文件中作为Y轴数据的列名。
    interpolation_step_size (float): 插值后x轴上的步长。
    kind (str): 插值方法。

    返回:
    list of dict: 包含每个CSV文件插值结果的列表。
                  每个结果是一个字典，包含 'x_interpolated', 'y_interpolated' 键，
                  以及 'segment_name' (CSV文件相对于 input_root_folder 的路径)。
    """
    
    all_segments_raw_data = []
    
    if not os.path.isdir(input_root_folder):
        print(f"错误: 指定的输入根文件夹 '{input_root_folder}' 不存在或不是一个目录。请检查路径。")
        return []

    # 使用 os.walk 遍历所有子目录和文件
    for dirpath, dirnames, filenames in os.walk(input_root_folder):
        for file_name in filenames:
            if file_name.lower().endswith('.csv'):
                csv_file_path = os.path.join(dirpath, file_name)
                # 获取CSV文件相对于输入根目录的路径，作为其唯一标识
                relative_path = os.path.relpath(csv_file_path, input_root_folder)
                
                print(f"正在处理文件: {relative_path}")
                
                try:
                    df = pd.read_csv(csv_file_path)
                    
                    # 检查列是否存在
                    if x_col_name in df.columns and y_col_name in df.columns:
                        # 确保数据是数值类型，并删除NaN值
                        x_values = pd.to_numeric(df[x_col_name], errors='coerce')
                        y_values = pd.to_numeric(df[y_col_name], errors='coerce')
                        
                        combined_df = pd.DataFrame({'x': x_values, 'y': y_values}).dropna()
                        
                        if not combined_df.empty:
                            all_segments_raw_data.append({
                                'x': combined_df['x'].tolist(),
                                'y': combined_df['y'].tolist(),
                                'segment_name': relative_path # 使用相对路径作为段名称
                            })
                        else:
                            print(f"    警告: 文件 '{relative_path}' 在指定列中没有有效数值数据。跳过。")
                    else:
                        print(f"    警告: 文件 '{relative_path}' 中缺少列 '{x_col_name}' 或 '{y_col_name}'。跳过。")
                except pd.errors.EmptyDataError:
                    print(f"    警告: 文件 '{relative_path}' 是空的。跳过。")
                except Exception as e:
                    print(f"    错误: 读取文件 '{relative_path}' 时发生错误: {e}. 跳过。")
    
    # 现在调用核心插值函数来处理收集到的所有段数据 (每个CSV是一个段)
    interpolated_results = interpolate_variable_segments_decimal(
        segments_data=all_segments_raw_data,
        interpolation_step_size=interpolation_step_size,
        kind=kind
    )
            
    return interpolated_results

# --- 多列插值处理函数 ---
def process_multiple_columns_interpolation(input_root_folder, x_col_name, y_col_names, interpolation_step_size=1.0, kind='cubic'):
    """
    对多个Y列进行插值，共享相同的X轴。
    
    参数:
    input_root_folder (str): 包含CSV文件的根目录路径。
    x_col_name (str): 作为X轴的列名。
    y_col_names (list): 需要插值的Y列名列表。
    interpolation_step_size (float): 插值步长。
    kind (str): 插值方法。
    
    返回:
    dict: 按文件组织的插值结果，格式为 {file_path: {col_name: {'x': x_data, 'y': y_data}}}
    """
    results = {}
    
    if not os.path.isdir(input_root_folder):
        print(f"错误: 指定的输入根文件夹 '{input_root_folder}' 不存在或不是一个目录。请检查路径。")
        return {}

    # 遍历所有CSV文件
    for dirpath, dirnames, filenames in os.walk(input_root_folder):
        for file_name in filenames:
            if file_name.lower().endswith('.csv'):
                csv_file_path = os.path.join(dirpath, file_name)
                relative_path = os.path.relpath(csv_file_path, input_root_folder)
                
                print(f"正在处理文件: {relative_path}")
                
                try:
                    df = pd.read_csv(csv_file_path)
                    
                    # 检查X列是否存在
                    if x_col_name not in df.columns:
                        print(f"    警告: 文件 '{relative_path}' 中缺少X列 '{x_col_name}'。跳过。")
                        continue
                    
                    # 检查哪些Y列存在
                    available_y_cols = [col for col in y_col_names if col in df.columns]
                    if not available_y_cols:
                        print(f"    警告: 文件 '{relative_path}' 中没有找到任何指定的Y列。跳过。")
                        continue
                    
                    results[relative_path] = {}
                    
                    # 对每个Y列进行插值
                    for y_col in available_y_cols:
                        print(f"    处理列: {y_col}")
                        
                        # 准备数据
                        x_values = pd.to_numeric(df[x_col_name], errors='coerce')
                        y_values = pd.to_numeric(df[y_col], errors='coerce')
                        
                        combined_df = pd.DataFrame({'x': x_values, 'y': y_values}).dropna()
                        
                        if combined_df.empty:
                            print(f"        警告: 列 '{y_col}' 没有有效数据。跳过。")
                            continue
                        
                        # 准备插值数据
                        segments_data = [{
                            'x': combined_df['x'].tolist(),
                            'y': combined_df['y'].tolist(),
                            'segment_name': f"{relative_path}_{y_col}"
                        }]
                        
                        # 执行插值
                        interpolated_results = interpolate_variable_segments_decimal(
                            segments_data=segments_data,
                            interpolation_step_size=interpolation_step_size,
                            kind=kind
                        )
                        
                        if interpolated_results and len(interpolated_results[0]['x_interpolated']) > 0:
                            results[relative_path][y_col] = {
                                'x': interpolated_results[0]['x_interpolated'],
                                'y': interpolated_results[0]['y_interpolated']
                            }
                        else:
                            print(f"        警告: 列 '{y_col}' 插值失败。")
                    
                except pd.errors.EmptyDataError:
                    print(f"    警告: 文件 '{relative_path}' 是空的。跳过。")
                except Exception as e:
                    print(f"    错误: 读取文件 '{relative_path}' 时发生错误: {e}. 跳过。")
    
    return results

# --- 保存多列插值结果并计算电阻的函数 ---
def save_interpolated_results_with_resistance(interpolated_data_dict, output_root_folder, x_col_name):
    """
    将多列插值结果保存到CSV文件，并计算电阻列（电压/电流）。
    
    参数:
    interpolated_data_dict (dict): 多列插值结果字典。
    output_root_folder (str): 输出根目录。
    x_col_name (str): X列的名称。
    """
    
    if not os.path.exists(output_root_folder):
        os.makedirs(output_root_folder)
        print(f"已创建输出目录: {os.path.abspath(output_root_folder)}")
    else:
        print(f"输出目录已存在: {os.path.abspath(output_root_folder)}")

    for relative_path, columns_data in interpolated_data_dict.items():
        if not columns_data:
            print(f"跳过保存文件 '{relative_path}'，因为它没有插值数据。")
            continue

        # 构建输出文件路径
        relative_dir = os.path.dirname(relative_path)
        file_name_without_ext, ext = os.path.splitext(os.path.basename(relative_path))
        
        output_dir_for_file = os.path.join(output_root_folder, relative_dir)
        os.makedirs(output_dir_for_file, exist_ok=True)
        
        output_file_name = f"{file_name_without_ext}_interpolated{ext}"
        output_file_path = os.path.join(output_dir_for_file, output_file_name)
        
        # 获取共同的X轴数据（所有列应该有相同的X轴）
        first_col = next(iter(columns_data.values()))
        x_data = first_col['x']
        
        # 构建输出DataFrame
        output_data = {x_col_name: x_data}
        
        # 添加所有插值后的Y列
        for col_name, col_data in columns_data.items():
            output_data[col_name] = col_data['y']
        
        # 计算电阻列（如果同时有电压和电流列）
        if 'max_cell_voltage' in columns_data and 'charge_current' in columns_data:
            voltage_data = columns_data['max_cell_voltage']['y']
            current_data = columns_data['charge_current']['y']
            
            # 计算电阻，避免除零错误
            resistance_data = np.where(current_data != 0, 
                                     voltage_data / current_data, 
                                     np.nan)
            output_data['resistance'] = resistance_data
            print(f"    已计算电阻列（电压/电流）")
        
        # 创建DataFrame并保存
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_file_path, index=False)
        print(f"已保存插值数据到: {output_file_path}")
        print(f"    包含列: {list(output_data.keys())}")


# --- 主程序执行流程 ---
if __name__ == "__main__":
    print("--- 多列数据插值工具 ---")
    print("该工具将对指定的多个Y列进行插值，并计算电阻（电压/电流）")
    print("--------------------")

    # 设置参数
    input_folder = 'charging_voltage_curves'
    output_folder = 'input_data'
    x_column = 'available_capacity'
    
    # 要插值的Y列
    y_columns = ['max_cell_voltage', 'charge_current', 'max_temperature','charging_duration_seconds','available_energy']
    
    interpolation_step_size = 1.0

    interpolation_kind = input("请输入插值方法 (linear, quadratic, cubic, 默认为 cubic): ").strip().lower()
    if interpolation_kind not in ['linear', 'quadratic', 'cubic']:
        print("无效的插值方法，将使用默认值 'cubic'。")
        interpolation_kind = 'cubic'

    print(f"\n--- 开始处理多列数据插值 ---")
    print(f"X轴列: {x_column}")
    print(f"Y轴列: {', '.join(y_columns)}")
    
    # 执行多列插值
    interpolated_data = process_multiple_columns_interpolation(
        input_root_folder=input_folder,
        x_col_name=x_column,
        y_col_names=y_columns,
        interpolation_step_size=interpolation_step_size,
        kind=interpolation_kind
    )

    if interpolated_data:
        print("\n--- 开始保存插值结果并计算电阻 ---")
        save_interpolated_results_with_resistance(
            interpolated_data_dict=interpolated_data,
            output_root_folder=output_folder,
            x_col_name=x_column
        )
        print("\n所有插值任务完成！")
        print("输出文件包含以下列:")
        print(f"  - {x_column} (X轴)")
        print(f"  - {' '.join(y_columns)} (插值后的Y列)")
        print("  - resistance (电压/电流，如果两列都存在)")
    else:
        print("\n没有生成任何插值数据。请检查输入路径和文件内容。")