#!/usr/bin/env python
from sensor_msgs.msg import LaserScan
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
from kobuki_msgs.msg import BumperEvent
from gazebo_msgs.msg import ModelState


class Scan_msg():

	def __init__(self):

		self.rays = np.zeros(10);
		self.dist = np.zeros(2);
		self.aim = np.zeros(2);
		
		self.sub = rospy.Subscriber('/ron/laser/scan', LaserScan, self.sort)
		self.sub2=rospy.Subscriber('/odom',Odometry,self.odom_callback)
		self.sub2=rospy.Subscriber("/mobile_base/events/bumper",BumperEvent,self.BumperEventCallback)

		# self.pub = rospy.Publisher('/ron/cmd_vel_mux',Twist,queue_size=10)
		self.pub = rospy.Publisher('/cmd_vel_mux/input/navi',Twist,queue_size=10)
		self.pub2 = rospy.Publisher("/gazebo/set_model_state",ModelState,queue_size=10);

		self.move_cmd = Twist()
		self.move_cmd.linear.x = 0.5
		self.move_cmd.angular.z = 0
		self.initState=ModelState()



		

	def sort(self, laserscan):
		index=0
		# num = len(laserscan.ranges)
		temp=np.array(laserscan.ranges)
		where_is_nans=np.isnan(temp);
		temp[where_is_nans]=7;
		self.rays=temp[180:539:36] # 0 left 720 right	
		# rospy.loginfo(self.rays)

	def odom_callback(self,msg):
		distance=np.sqrt((msg.pose.pose.position.x-(self.aim[0]))**2+(msg.pose.pose.position.y-(self.aim[1]))**2)
		angle=np.arctan2((msg.pose.pose.position.y-(self.aim[1])),(msg.pose.pose.position.x-(self.aim[0])))
		self.dist=np.array([distance,angle])
		# rospy.loginfo(self.aim)
		self.move()

	def move(self):
		self.pub.publish(self.move_cmd)

	def BumperEventCallback(self,data):
		self.move_cmd.linear.x = 0.0
		self.move_cmd.angular.z = 0
		rospy.loginfo("Im hit!")
		self.resetPose();

	def resetPose(self):
		index=np.random.randint(4)
		start=np.array([[0,4,0],[-4,0,0],[0,-4,0],[4,0,0]])
		goals=np.array([[-4,0,0],[4,0,0],[0,4,0],[0,-4,0]])
		self.initState.model_name = "mobile_base";
  		self.initState.reference_frame = "world";
	  	self.initState.pose.position.x = start[index][0]
	  	self.initState.pose.position.y = start[index][1]
	 	self.initState.pose.orientation.z=0;
	 	self.initState.pose.orientation.w=0;
	 	self.aim=[goals[index][0],goals[index][1]]
  		self.pub2.publish(self.initState);
  		self.move_cmd.linear.x = 0.5
		self.move_cmd.angular.z = 0



def listener():
	rospy.init_node('navigation_sensors')
	rospy.loginfo("Subscriber Starting")
	
	rospy.spin()

if __name__ == '__main__':
	try:
		Scan_msg()
		listener()
	except rospy.ROSInterruptException:
		rospy.loginfo("exception")
