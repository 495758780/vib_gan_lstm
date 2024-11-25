import numpy as np
import pandas as pd

# 参数数量和组数
num_parameters = 20  # 每组有 20 个参数
num_groups = 20      # 生成 20 组

# 随机生成工艺参数（假设范围在[0, 100]之间）
parameter_data = np.random.randint(10, 100, size=(num_groups, num_parameters))

# 转换为 DataFrame 便于查看
columns = [f"Param_{i+1}" for i in range(num_parameters)]
parameter_df = pd.DataFrame(parameter_data, columns=columns)

# 保存为 CSV 文件
parameter_df.to_csv("process_parameters.csv", index=False)

# 打印生成的数据
print(parameter_df)
