#!/usr/bin/env python
from sensor_msgs.msg import LaserScan
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
from kobuki_msgs.msg import BumperEvent
from gazebo_msgs.msg import ModelState
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Float32
from std_msgs.msg import Empty
from rospy.numpy_msg import numpy_msg
from taxi.msg import Floats


class Scan_msg():

	def __init__(self):

		self.rays = np.zeros(10);
		self.dist = np.zeros(2);
		self.aim = np.array([2,2]);
		self.pause=0
		self.move_cmd = Twist()
		self.move_cmd.linear.x = 0.5
		self.move_cmd.angular.z = 0
		self.overwrite=0
		
		self.sub = rospy.Subscriber('/ron/laser/scan', LaserScan, self.sort)
		self.sub2=rospy.Subscriber('/odom',Odometry,self.odom_callback)
		self.sub3=rospy.Subscriber("/mobile_base/events/bumper",BumperEvent,self.BumperEventCallback)
		# self.sub4 = rospy.Subscriber("/gazebo/set_model_state",ModelState,self.odom_callback);

		# self.pub = rospy.Publisher('/ron/cmd_vel_mux',Twist,queue_size=10)
		# self.pub = rospy.Publisher('/cmd_vel_mux/input/navi',Twist,queue_size=10)
		self.pub2 = rospy.Publisher("/gazebo/set_model_state",ModelState,queue_size=10);
		self.pub3 = rospy.Publisher('state', Floats,queue_size=10)
		self.pub4 = rospy.Publisher('/mobile_base/commands/reset_odometry',Empty,queue_size=10)

		
		self.initState=ModelState()
		# print np.siz
		
		rospy.Timer(rospy.Duration(1),self.loop)



		

	def sort(self, laserscan):
		# num = len(laserscan.ranges)
		temp=np.array(laserscan.ranges)
		where_is_nans=np.isnan(temp);
		temp[where_is_nans]=7;
		self.rays=temp[180:539:36] # 0 left 720 right	
		# rospy.loginfo(self.rays)

	def odom_callback(self,msg):
		# distance=np.sqrt((msg.pose.pose.position.x-(self.aim[0]))**2+(msg.pose.pose.position.y-(self.aim[1]))**2)
		distance=np.sqrt((msg.pose.pose.position.x-self.aim[0])**2+(msg.pose.pose.position.y-self.aim[1])**2)
		angle=np.arctan2((msg.pose.pose.position.y-(self.aim[1])),(msg.pose.pose.position.x-(self.aim[0])))
		self.dist=np.array([distance,angle])
		# rospy.loginfo(self.dist)
		# self.move()

	def move(self):
		self.pub.publish(self.move_cmd)

	def BumperEventCallback(self,data):
		self.move_cmd.linear.x = 0.0
		self.move_cmd.angular.z = 0
		rospy.loginfo("Im hit!")
		self.pause=1;
		self.move_cmd.linear.x = 0.5
		self.move_cmd.angular.z = 0
		self.resetPose();
		self.overwrite=1

	def resetPose(self):
		index=np.random.randint(4)
		# start=np.array([[0,4,0],[-4,0,0],[0,-4,0],[4,0,0]])
		goals=np.array([[2,2,0],[-3,2,0],[2,-2,0],[-3,-2,0]])
		self.initState.model_name = "mobile_base";
		self.initState.reference_frame = "world";
		# self.initState.pose.position.x = start[index][0]
		# self.initState.pose.position.y = start[index][1]
		self.initState.pose.position.x=0
		self.initState.pose.position.y=0
		z=np.random.uniform(0, 1)
		self.initState.pose.orientation.z=z
		self.initState.pose.orientation.w=np.sqrt(1-z**2);
		self.aim=[goals[index][0],goals[index][1]] #brings everything to world frame
		self.pub2.publish(self.initState);
		self.pub4.publish()
		self.move_cmd.linear.x = 0.5
		self.move_cmd.angular.z = 0
		self.pause=0;

	def loop(self,event):
		if self.pause==0:
			self.outputs=np.concatenate((self.rays,self.dist))
			if self.overwrite==1:
				self.outputs[10]=1234
				self.overwrite=0
			self.pub3.publish(self.outputs)

		# rospy.loginfo(self.outputs)
		# rospy.loginfo("hrjh")



def listener():
	rospy.init_node('navigation_sensors')
	rospy.loginfo("Subscriber Starting")
	Scan_msg()
	rospy.spin()

if __name__ == '__main__':
	try:
		listener()
		
		
	except rospy.ROSInterruptException:
		rospy.loginfo("exception")
