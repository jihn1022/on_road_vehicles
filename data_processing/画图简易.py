import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('interpolated_dqdv/#1/interpolated_charging_session_002_20190727_040656.csv')

# 获取第一列和第二列数据
x_data = df.iloc[:, 0]  # 第一列
y_data = df.iloc[:, 2]  # 第二列

# 创建图形
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, marker='o', linestyle='-', linewidth=2, markersize=4)

# 设置图形标题和轴标签
plt.title('x,y Data Plot')
plt.xlabel(df.columns[0])  # 使用第一列的列名作为x轴标签
plt.ylabel(df.columns[1])  # 使用第二列的列名作为y轴标签

# 添加网格
plt.grid(True, alpha=0.3)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()