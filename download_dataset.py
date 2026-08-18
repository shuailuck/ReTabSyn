import pandas as pd
from sklearn.datasets import fetch_openml

# 1. 加载数据
adult = fetch_openml(name='adult', version=2, as_frame=True)

# 2. 合并特征与目标为一个完整 Dataframe
df = adult.frame  # fetch_openml 提供了 .frame 属性，包含 X 和 y

# 3. 导出保存到本地
df.to_csv('./csv/example/adult_dataset.csv', index=False)
# 或者导出为更高效的 Parquet 格式
# df.to_parquet('adult_dataset.parquet', index=False)

print("数据集已成功保存到本地 adult_dataset.csv！")