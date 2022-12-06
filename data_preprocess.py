import os
import pandas as pd
import torch

#创建一个人工dataset，存储在csv中
os.makedirs(os.path.join('.', 'data'), exist_ok=True)
data_file = os.path.join('.', 'data', 'house_tiny.csv')

with open(data_file, 'w') as f:
    f.write('NumRooms, Alley, Price\n')#列名
    f.write('NA, Pave, 127500\n')#每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')  

data = pd.read_csv(data_file)
"""
print(data)
   NumRooms  Alley   Price
0       NaN   Pave  127500
1       2.0    NaN  106000
2       4.0    NaN  178100
3       NaN    NaN  140000
"""
#处理缺失数据，插值或者删除
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
# print(inputs) 
inputs = pd.get_dummies(inputs, dummy_na=True)#类似于one-hot编码
# print(inputs)

x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x, y)
