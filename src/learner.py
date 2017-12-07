import rospy
from taxi.msg import Floats
import numpy as np
import torch

# from moveCNN import *
dtype = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor

class stateMsg():

	def __init__(self):

		self.state = np.zeros(14);
		self.stateT=torch.cuda.FloatTensor()
		self.sub = rospy.Subscriber('/state', Floats, self.fetch)


	def fetch(self,msg):
		self.state=np.array(msg.data)
		self.stateT=torch.from_numpy(self.state).type(dtype)
		print self.stateT
		# moveCNN(self.stateT)

def learner():
	rospy.init_node('learner')
	rospy.loginfo("Subscriber Starting")
	stateMsg()
	rospy.spin()

if __name__ == '__main__':
	try:
		learner()
		
		
	except rospy.ROSInterruptException:
		rospy.loginfo("exception")