import pandas as pd
import os

def count_data_points_in_folder(root_folder):
    """
    统计文件夹及子文件夹中所有CSV文件的数据点数量
    """
    total_files = 0
    total_data_points = 0
    file_details = []
    folder_stats = {}
    
    print(f"开始统计文件夹: {root_folder}")
    print("=" * 60)
    
    # 遍历所有子文件夹
    for root, dirs, files in os.walk(root_folder):
        folder_files = 0
        folder_points = 0
        
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                
                try:
                    # 读取CSV文件
                    df = pd.read_csv(file_path)
                    rows, cols = df.shape
                    data_points = rows * cols
                    
                    # 统计信息
                    file_info = {
                        'file_path': file_path,
                        'relative_path': os.path.relpath(file_path, root_folder),
                        'folder': os.path.relpath(root, root_folder) if root != root_folder else '根目录',
                        'rows': rows,
                        'columns': cols,
                        'data_points': data_points,
                        'column_names': list(df.columns)
                    }
                    
                    file_details.append(file_info)
                    
                    # 累计统计
                    total_files += 1
                    total_data_points += data_points
                    folder_files += 1
                    folder_points += data_points
                    
                    print(f"✓ {file_info['relative_path']}: {rows}行 × {cols}列 = {data_points}个数据点")
                    
                except Exception as e:
                    print(f"✗ 读取文件 {file} 时出错: {str(e)}")
                    continue
        
        # 记录文件夹统计
        if folder_files > 0:
            folder_rel_path = os.path.relpath(root, root_folder) if root != root_folder else '根目录'
            folder_stats[folder_rel_path] = {
                'files': folder_files,
                'data_points': folder_points
            }
    
    return total_files, total_data_points, file_details, folder_stats

def print_summary(total_files, total_data_points, file_details, folder_stats, root_folder):
    """
    打印统计摘要
    """
    print("\n" + "=" * 60)
    print("统计摘要")
    print("=" * 60)
    print(f"总文件数: {total_files}")
    print(f"总数据点数: {total_data_points:,}")
    print(f"平均每个文件数据点数: {total_data_points/total_files:.1f}" if total_files > 0 else "平均每个文件数据点数: 0")
    
    print(f"\n按文件夹统计:")
    for folder, stats in folder_stats.items():
        print(f"  {folder}: {stats['files']}个文件, {stats['data_points']:,}个数据点")
    
    # 按数据点数量排序显示前10个文件
    if file_details:
        print(f"\n数据点最多的前10个文件:")
        sorted_files = sorted(file_details, key=lambda x: x['data_points'], reverse=True)
        for i, file_info in enumerate(sorted_files[:10], 1):
            print(f"  {i}. {file_info['relative_path']}: {file_info['data_points']:,}个数据点 "
                  f"({file_info['rows']}行 × {file_info['columns']}列)")
    
    # 统计列数分布
    if file_details:
        column_counts = {}
        for file_info in file_details:
            cols = file_info['columns']
            column_counts[cols] = column_counts.get(cols, 0) + 1
        
        print(f"\n按列数分布:")
        for cols in sorted(column_counts.keys()):
            print(f"  {cols}列: {column_counts[cols]}个文件")

def save_detailed_report(file_details, folder_stats, total_files, total_data_points, root_folder, output_file=None):
    """
    保存详细报告到文件
    """
    if output_file is None:
        output_file = os.path.join(root_folder, 'data_points_report.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("CSV文件数据点统计详细报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"统计路径: {root_folder}\n")
        f.write(f"统计时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("总体统计:\n")
        f.write(f"  总文件数: {total_files}\n")
        f.write(f"  总数据点数: {total_data_points:,}\n")
        f.write(f"  平均每个文件数据点数: {total_data_points/total_files:.1f}\n\n" if total_files > 0 else "  平均每个文件数据点数: 0\n\n")
        
        f.write("按文件夹统计:\n")
        for folder, stats in folder_stats.items():
            f.write(f"  {folder}: {stats['files']}个文件, {stats['data_points']:,}个数据点\n")
        f.write("\n")
        
        f.write("详细文件列表:\n")
        f.write("-" * 60 + "\n")
        for file_info in sorted(file_details, key=lambda x: x['relative_path']):
            f.write(f"文件: {file_info['relative_path']}\n")
            f.write(f"  路径: {file_info['file_path']}\n")
            f.write(f"  所属文件夹: {file_info['folder']}\n")
            f.write(f"  行数: {file_info['rows']}\n")
            f.write(f"  列数: {file_info['columns']}\n")
            f.write(f"  数据点数: {file_info['data_points']:,}\n")
            f.write(f"  列名: {', '.join(file_info['column_names'])}\n")
            f.write("\n")
    
    print(f"详细报告已保存到: {output_file}")

# 设置要统计的文件夹路径
folder_path = '/home/jihn/battery-charging-data-of-on-road-electric-vehicles/processed_dqdv'

# 执行统计
total_files, total_data_points, file_details, folder_stats = count_data_points_in_folder(folder_path)

# 打印摘要
print_summary(total_files, total_data_points, file_details, folder_stats, folder_path)

# 保存详细报告
save_detailed_report(file_details, folder_stats, total_files, total_data_points, folder_path)

# 如果需要按特定列统计数据点，可以使用以下代码
print(f"\n" + "=" * 60)
print("按列统计非空数据点 (前3个文件示例):")
print("=" * 60)

for i, file_info in enumerate(file_details[:3]):
    try:
        df = pd.read_csv(file_info['file_path'])
        print(f"\n文件: {file_info['relative_path']}")
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            print(f"  {col}: {non_null_count}/{len(df)} 个非空数据点")
    except:
        continue

print(f"\n完成! 共统计了 {total_files} 个CSV文件，总计 {total_data_points:,} 个数据点")