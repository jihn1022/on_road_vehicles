import pandas as pd
import os
import numpy as np

def check_missing_values_in_csv_files(folder_path):
    """
    遍历文件夹中的所有CSV文件，检查缺失值
    
    Args:
        folder_path (str): 文件夹路径
    """
    # 存储所有文件的缺失值信息
    missing_data_summary = {}
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            
            try:
                # 读取CSV文件
                df = pd.read_csv(file_path)
                
                # 检查缺失值
                missing_info = {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'missing_by_column': df.isnull().sum().to_dict(),
                    'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                    'total_missing': df.isnull().sum().sum(),
                    'missing_percentage_overall': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
                }
                
                missing_data_summary[filename] = missing_info
                
                # 打印每个文件的缺失值信息
                print(f"\n{'='*50}")
                print(f"文件: {filename}")
                print(f"{'='*50}")
                print(f"总行数: {missing_info['total_rows']}")
                print(f"总列数: {missing_info['total_columns']}")
                print(f"总缺失值: {missing_info['total_missing']}")
                print(f"整体缺失率: {missing_info['missing_percentage_overall']:.2f}%")
                
                print("\n各列缺失值情况:")
                for col, missing_count in missing_info['missing_by_column'].items():
                    if missing_count > 0:
                        percentage = missing_info['missing_percentage'][col]
                        print(f"  {col}: {missing_count} ({percentage:.2f}%)")
                
                # 如果没有缺失值
                if missing_info['total_missing'] == 0:
                    print("  无缺失值")
                    
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {str(e)}")
    
    return missing_data_summary

def generate_missing_values_report(folder_path, output_file='missing_values_report.txt'):
    """
    生成缺失值报告并保存到文件
    
    Args:
        folder_path (str): 文件夹路径
        output_file (str): 输出报告文件名
    """
    missing_summary = check_missing_values_in_csv_files(folder_path)
    
    # 生成报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("CSV文件缺失值检查报告\n")
        f.write("="*60 + "\n\n")
        
        for filename, info in missing_summary.items():
            f.write(f"文件: {filename}\n")
            f.write("-" * 40 + "\n")
            f.write(f"总行数: {info['total_rows']}\n")
            f.write(f"总列数: {info['total_columns']}\n")
            f.write(f"总缺失值: {info['total_missing']}\n")
            f.write(f"整体缺失率: {info['missing_percentage_overall']:.2f}%\n\n")
            
            f.write("各列缺失值情况:\n")
            for col, missing_count in info['missing_by_column'].items():
                if missing_count > 0:
                    percentage = info['missing_percentage'][col]
                    f.write(f"  {col}: {missing_count} ({percentage:.2f}%)\n")
            
            if info['total_missing'] == 0:
                f.write("  无缺失值\n")
            
            f.write("\n" + "="*60 + "\n\n")
    
    print(f"\n报告已保存到: {output_file}")

# 使用示例
if __name__ == "__main__":
    # 设置要检查的文件夹路径
    folder_path = "/home/jihn/battery-charging-data-of-on-road-electric-vehicles/standardized_features"
    
    # 检查缺失值
    print("开始检查CSV文件中的缺失值...")
    missing_summary = check_missing_values_in_csv_files(folder_path)
    
    # 生成报告
    generate_missing_values_report(folder_path)
    
    # 显示汇总信息
    print(f"\n{'='*50}")
    print("汇总信息")
    print(f"{'='*50}")
    print(f"共检查了 {len(missing_summary)} 个CSV文件")
    
    files_with_missing = sum(1 for info in missing_summary.values() if info['total_missing'] > 0)
    print(f"有缺失值的文件数: {files_with_missing}")
    print(f"无缺失值的文件数: {len(missing_summary) - files_with_missing}")