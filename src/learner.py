import rospy
from taxi.msg import Floats
import numpy as np
import torch
from geometry_msgs.msg import Twist
from replayBuffer import ReplayBuffer
from moveCNN import *
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
# moveNet = moveCNN()
actor=actorNet()
critic=criticNet()
trainsteps=20


class stateMsg():

	def __init__(self):

		# self.moveNet = moveCNN()
		self.state = np.zeros(14);
		self.lstate = np.zeros(14);
		self.stateT=torch.FloatTensor()
		self.sub = rospy.Subscriber('/state', Floats, self.fetch)
		self.pub = rospy.Publisher('/cmd_vel_mux/input/navi',Twist,queue_size=10)
		self.fpass=1
		self.move_cmd = Twist()
		self.move_cmd.linear.x = 0
		self.move_cmd.angular.z = 0
		self.max_episodes=20000
		self.episode_length=20 #maybe change later
		self.num_episodes=0
		self.terminal=0
		self.rBuf=ReplayBuffer()

	def train(self,states):
		states=torch.unsqueeze(states, 0)
		X = Variable(states.clone().cpu())
		actions=actor.forward(X)
		Q=critic.forward(X,actions)
		return Q, actions


	def fetch(self,msg):
		if self.num_episodes<self.max_episodes:
			self.state=np.array(msg.data)
			self.state=np.concatenate((self.state,np.array([self.move_cmd.linear.x,self.move_cmd.angular.z])))
			self.stateT=torch.from_numpy(self.state).type(dtype)
			# Q,actions=self.train(self.stateT)
			states=torch.unsqueeze(self.stateT, 0)
			X = Variable(states.clone().cpu())
			print X
			actions=actor.forward(X)
			action=actions.data.numpy()
			self.move_cmd.linear.x = action[0][0]
			self.move_cmd.angular.z = action[0][1]
			self.pub.publish(self.move_cmd)
			if self.fpass==0:
				R=self.reward()
				self.rBuf.add(self.lstate,action,R,self.terminal,self.state)
				if self.rBuf.size()>5:
					s_batch, a_batch, r_batch, t_batch, s2_batch = self.rBuf.sample_batch(5)
					
				# Q=critic.forward(X,actions)
			if self.fpass==1:
				self.fpass=0	
			self.lstate=self.state

	def reward(self):
		dist=self.state[10]
		ldist=self.lstate[10]
		# print dist
		if dist<0.2:
			R=10
			self.terminal=1
			self.num_episodes+=1
		elif dist==1234:
			R=-100
			self.terminal=1
			self.num_episodes+=1
			# print "hit"
		else:
			R=0.1*(ldist-dist)

		return R



def learner():
	rospy.init_node('learner')
	# rospy.loginfo("Subscriber Starting")
	stateMsg()
	rospy.spin()

if __name__ == '__main__':
	try:
		learner()
		
		
	except rospy.ROSInterruptException:
		rospy.loginfo("exception")