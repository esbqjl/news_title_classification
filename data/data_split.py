import random

# 读取数据文件
with open('./data/toutiao_cat_data.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 计算每个数据集的大小
total_lines = len(lines)
train_size = int(total_lines * 0.7)
dev_size = int(total_lines * 0.15)
# The rest will go to the test set

# 打乱数据顺序
random.shuffle(lines)

# 分配数据
train_data = lines[:train_size]
dev_data = lines[train_size:train_size+dev_size]
test_data = lines[train_size+dev_size:]

# 保存到新文件
with open('./data/toutiao_cat_data.train.txt', 'w', encoding='utf-8') as file:
    file.writelines(train_data)

with open('./data/toutiao_cat_data.dev.txt', 'w', encoding='utf-8') as file:
    file.writelines(dev_data)

with open('./data/toutiao_cat_data.test.txt', 'w', encoding='utf-8') as file:
    file.writelines(test_data)

# 输出数据集的大小作为确认
print(f"Train set: {len(train_data)} lines")
print(f"Dev set: {len(dev_data)} lines")
print(f"Test set: {len(test_data)} lines")