import rospy
from taxi.msg import Floats
import numpy as np
import torch
from geometry_msgs.msg import Twist

from moveCNN import *
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
# moveNet = moveCNN()
actor=actorNet()
critic=criticNet()

class stateMsg():

	def __init__(self):

		# self.moveNet = moveCNN()
		self.state = np.zeros(14);
		self.stateT=torch.cuda.FloatTensor()
		self.sub = rospy.Subscriber('/state', Floats, self.fetch)
		self.pub = rospy.Publisher('/cmd_vel_mux/input/navi',Twist,queue_size=10)

		self.move_cmd = Twist()
		self.move_cmd.linear.x = 0
		self.move_cmd.angular.z = 0

	def train(self,states):
		states=torch.unsqueeze(states, 0)
		X = Variable(states.clone().cpu())
		# self.actorForward(X)		
		# self.criticForward(X)
		actions=actor.forward(X)
		Q=critic.forward(X,actions)

		# print self.actions
		# return self.actions
		return Q, actions


	def fetch(self,msg):
		self.state=np.array(msg.data)		
		self.state=np.concatenate((self.state,np.array([self.move_cmd.linear.x,self.move_cmd.angular.z])))
		self.stateT=torch.from_numpy(self.state).type(dtype)
		# print type(self.stateT)
		Q,actions=self.train(self.stateT)
		# print actions[0][0]
		actions=actions.data.numpy()
		self.move_cmd.linear.x = actions[0][0]
		self.move_cmd.angular.z = actions[0][1]
		self.pub.publish(self.move_cmd)


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