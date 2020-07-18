import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
 
x = torch.unsqueeze(torch.linspace(-1,1,100),dim = 1)
#unsqueeze函数创建一个维度是1的向量
#linspace函数生成在-1到1之间的100个数
 
y = x**2+0.2*torch.rand(x.size())
#得到x自乘的矩阵，然后加上同x矩阵相同的噪声
 
print(x,y,x.size())
#输出x，y矩阵，以及矩阵x的大小
 
#以上操作为初始化矩阵
 
x,y = Variable(x),Variable(y)
#将矩阵转化为 变量
 
class Net(torch.nn.Module):
#定义神经网络
	def __init__(self,n_feature,n_hidden,n_output):
	#初始化数组，参数分别是初始化信息，特征数，隐藏单元数，输出单元数
		super(Net,self).__init__()
		#此步骤是官方要求
                self.hidden = torch.nn.Linear(n_feature,n_hidden)
		#设置输入层到隐藏层的函数
		self.predict = torch.nn.Linear(n_hidden,n_output)
		#设置隐藏层到输出层的函数
 
	def forward(self,x):
	#定义向前传播函数
		x = F.relu(self.hidden(x))
        #给x加权成为a，用激励函数将a变成特征b
		x = self.predict(x)
        #给b加权，预测最终结果
		return x
net = Net(1,10,1)
#定义一个神经网络
print(net)
#查看各层之间的参数

