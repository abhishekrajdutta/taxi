import torch.optim as optim
from torch.autograd import Variable
from chainer import cuda, optimizers, serializers
from chainer import Variable as vb
import chainer.functions as F
import math
import numpy as np

dtype = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor

class moveCNN(object):
	def __init__(self,norm):
        super(ResCNN, self).__init__()
		
		self.actions=torch.FloatTensor()
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

        
        #critic network
		self.clin1 = denseLayer(14, 512)
        self.cin1 = torch.nn.InstanceNorm2d(512, affine=True)
        self.clin2 = denseLayer(514, 512)
        self.cin2 = torch.nn.InstanceNorm2d(512, affine=True)
        self.clin3 = denseLayer(512, 512)
        self.cin3 = torch.nn.InstanceNorm2d(512, affine=True)

        self.clinQ = denseLayer(512, 1)
        self.inQ = torch.nn.InstanceNorm2d(1, affine=True)
        
        # Non-linearities
        self.relu = torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.Tanh=torch.nn.Tanh()

    def actorForward(self, states):
    	X = Variable(states.clone().type(dtype))

    	#Forward pass on actor network
    	ay=self.ReLU(self.ain1(self.alin1(X)))
    	ay=self.ReLU(self.ain2(self.alin2(ay)))
    	ay=self.ReLU(self.ain3(self.alin3(ay)))
    	ayLinear=self.Sigmoid(self.ainLinear(self.alinLinear(ay)))
    	ayAngular=self.Sigmoid(self.ainLinear(self.alinAngular(ay)))
    	self.actions=torch.cat((ayLinear,ayAngular),0).type(dtype)

    def criticForward(self,states)	
    	#Forward pass on critic network
    	cy=self.ReLU(self.cin1(self.clin1(X)))
    	cy=torch.cat((cy,self.actions),0)
    	cy=self.ReLU(self.cin2(self.clin2(cy)))
    	cy=self.ReLU(self.cin3(self.clin3(cy)))
    	Q=self.ReLU(self.cinQ(self.clinQ(cy)))

class denseLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(denseLayer, self).__init__()
        self.linear = torch.nn.linear(in_channels, out_channels)

    def forward(self, x):
        out = self.linear(out)
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
