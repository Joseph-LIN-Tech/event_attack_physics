import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# class STEFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         return (input > 0.5).float()

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output

# class STEBinarizer(nn.Module):
#     def __init__(self):
#         super(STEBinarizer, self).__init__()

#     def forward(self, x):
#         return STEFunction.apply(x)

# class ChessboardNet(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(ChessboardNet, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#         self.binarizer = STEBinarizer()

#     def forward(self, x):
#         print("x.shape",x.shape)
#         x = F.relu(self.fc1(x))
#         x = torch.sigmoid(self.fc2(x))
#         x = self.binarizer(x)
        
#         # 将输出reshape成棋盘格形式
#         x = x.view(-1, 5, 5)  # 将输出reshape为5x5的棋盘格形式
#         x = x.repeat_interleave(20, dim=1).repeat_interleave(20, dim=2)  # 扩展到100x100
#         return x

# # 生成UV map
# input_dim = 25  # 输入维度
# hidden_dim = 64  # 隐藏层维度
# output_dim = 25  # 输出维度，与输入相同

# # 实例化网络
# net = ChessboardNet(input_dim, hidden_dim, output_dim).cuda()

# # 生成随机输入（0或1）
# input_data = torch.randint(0, 2, (1, 25)).float().cuda()

# # 前向传播
# uv_map = net(input_data)

# # 打印UV map的形状
# print("UV Map Shape:", uv_map.shape)

# # 可视化并保存UV map
# plt.imshow(uv_map.detach().cpu().numpy()[0], cmap='gray')
# plt.title("Generated UV Map")
# plt.savefig('/home/lgx/code/AAAI25/Attack/attack_event_NMI_fullmask_obj_avg_chess/utils/generated_uv_map.png')
# plt.show()
