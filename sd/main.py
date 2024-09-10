import torch

# 创建一个形状为 [2, 3, 4] 的张量
input = torch.tensor(
    [
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
    ]
)

# 创建一个形状为 [2, 3] 的索引张量
index = torch.tensor([[[0, 1, 2]], [[3, 2, 1]]])  # 第一个批次的索引  # 第二个批次的索引

result = torch.gather(input=input, index=index, dim=-1)
tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print((1,) * 3)
