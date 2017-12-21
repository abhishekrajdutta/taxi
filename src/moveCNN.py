import torch.optim as optim
from torch.autograd import Variable
# from chainer import cuda, optimizers, serializers
# from chainer import Variable as vb
# import chainer.functions as F
import math
import numpy as np
import torch

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class actorNet(torch.nn.Module):
	def __init__(self):
		super(actorNet, self).__init__()
		#actor network    	
		self.alin1 = denseLayer(14, 512)
		self.ain1 = torch.nn.InstanceNorm2d(512, affine=True)
		self.alin2 = denseLayer(512, 512)
		self.ain2 = torch.nn.InstanceNorm2d(512, affine=True)
		self.alin3 = denseLayer(512, 512)        
		self.ain3 = torch.nn.InstanceNorm2d(512, affine=True)

		self.alinLinear = denseLayer(512, 1)
		self.ainLinear = torch.nn.InstanceNorm2d(1, affine=True)
		self.alinAngular = denseLayer(512, 1)
		self.ain2Angular = torch.nn.InstanceNorm2d(1, affine=True)

		# Non-linearities
		
		self.Sigmoid=torch.nn.Sigmoid()
		self.Tanh=torch.nn.Tanh()
		self.ReLU = torch.nn.ReLU()


	def forward(self, X):
		#Forward pass on actor network
		ay=self.ReLU(self.ain1(self.alin1(X)))
		ay=self.ReLU(self.ain2(self.alin2(ay)))
		ay=self.ReLU(self.ain3(self.alin3(ay)))
		ayLinear=self.Sigmoid(self.ainLinear(self.alinLinear(ay)))
		ayAngular=self.Sigmoid(self.ainLinear(self.alinAngular(ay)))
		ay=torch.cat((ayLinear,ayAngular),1)
		return ay

class criticNet(torch.nn.Module):
	def __init__(self):
		super(criticNet, self).__init__()

		#critic network
		self.clin1 = denseLayer(14, 512)
		self.cin1 = torch.nn.InstanceNorm2d(512, affine=True)
		self.clin2 = denseLayer(514, 512)
		self.cin2 = torch.nn.InstanceNorm2d(512, affine=True)
		self.clin3 = denseLayer(512, 512)
		self.cin3 = torch.nn.InstanceNorm2d(512, affine=True)

		self.clinQ = denseLayer(512, 1)
		self.cinQ = torch.nn.InstanceNorm2d(1, affine=True)

		self.ReLU = torch.nn.ReLU()

	def forward(self,X,actions):
		#Forward pass on critic network
		cy=self.ReLU(self.cin1(self.clin1(X)))
		cy=torch.cat((cy,actions),1)
		cy=self.ReLU(self.cin2(self.clin2(cy)))
		cy=self.ReLU(self.cin3(self.clin3(cy)))
		cy=self.ReLU(self.cinQ(self.clinQ(cy))).type(dtype)
		return cy



class denseLayer(torch.nn.Module):
	def __init__(self, in_channels, out_channels):
		super(denseLayer, self).__init__()
		self.linear = torch.nn.Linear(in_channels, out_channels)

	def forward(self, x):
		out = self.linear(x)
		return out


class ResidualBlock(torch.nn.Module):
	"""ResidualBlock
	introduced in: https://arxiv.org/abs/1512.03385
	recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
	"""

	def __init__(self, channels):
		super(ResidualBlock, self).__init__()
		self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
		self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
		self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
		self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
		self.relu = torch.nn.ReLU()

	def forward(self, x):
		residual = x
		out = self.relu(self.in1(self.conv1(x)))
		out = self.in2(self.conv2(out))
		out = out + residual
		return out
