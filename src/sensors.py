#!/usr/bin/env python
from sensor_msgs.msg import LaserScan
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
class Scan_msg():

	def __init__(self):

		# self.pub = rospy.Publisher('/ron/cmd_vel_mux',Twist,queue_size=10)
		self.pub = rospy.Publisher('/cmd_vel_mux/input/navi',Twist,queue_size=10)

		self.rays = np.zeros(5);
		self.aim=0;
		
		self.sub = rospy.Subscriber('/ron/laser/scan', LaserScan, self.sort)
		# self.sub2=rospy.Subscriber('/ron/odom_diffdrive',Odometry,self.odom_callback)
		# self.sub = rospy.Subscriber('/scan', LaserScan, self.sort)
		self.sub2=rospy.Subscriber('/odom',Odometry,self.odom_callback)
		self.move_cmd = Twist()
		

	def sort(self, laserscan):
		index=0
		# num = len(laserscan.ranges)
		temp=np.array(laserscan.ranges)
		where_is_nans=np.isnan(temp);
		temp[where_is_nans]=7;
		self.rays=temp[180:539:36] # 0 left 720 right	
		rospy.loginfo(self.rays)

	def odom_callback(self,msg):
		distance=np.sqrt((msg.pose.pose.position.x-(-4))**2+(msg.pose.pose.position.y-(0))**2)
		angle=np.arctan2(msg.pose.pose.position.y,msg.pose.pose.position.x)
		self.aim=np.array([distance,angle])
		# rospy.loginfo(self.aim)
		self.move()

	def move(self):
		self.move_cmd.linear.x = 0.0
		self.move_cmd.angular.z = 0
		self.pub.publish(self.move_cmd)


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
