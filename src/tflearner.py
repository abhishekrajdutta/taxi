import rospy
from taxi.msg import Floats
from std_msgs.msg import Empty
import numpy as np
from geometry_msgs.msg import Twist
from replayBuffer import ReplayBuffer
# from moveCNN import *

import tensorflow as tf
import tflearn
import argparse
import pprint as pp


class ActorNetwork(object):
	"""
	Input to the network is the state, output is the action
	under a deterministic policy.
	The output layer activation is a tanh to keep the action
	between -action_bound and action_bound
	"""

	def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
		self.sess = sess
		self.s_dim = state_dim
		self.a_dim = action_dim
		self.action_bound = action_bound
		self.learning_rate = learning_rate
		self.tau = tau
		self.batch_size = batch_size

		# Actor Network
		self.inputs, self.out, self.scaled_out = self.create_actor_network()

		self.network_params = tf.trainable_variables()

		# Target Network
		self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

		self.target_network_params = tf.trainable_variables()[
			len(self.network_params):]

		# Op for periodically updating target network with online network
		# weights
		self.update_target_network_params = \
			[self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
												  tf.multiply(self.target_network_params[i], 1. - self.tau))
				for i in range(len(self.target_network_params))]

		# This gradient will be provided by the critic network
		self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

		# Combine the gradients here
		self.unnormalized_actor_gradients = tf.gradients(
			self.scaled_out, self.network_params, -self.action_gradient)
		self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

		# Optimization Op
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
			apply_gradients(zip(self.actor_gradients, self.network_params))

		self.num_trainable_vars = len(
			self.network_params) + len(self.target_network_params)

	def create_actor_network(self):
		inputs = tflearn.input_data(shape=[None, self.s_dim])
		net = tflearn.fully_connected(inputs, 512)
		net = tflearn.layers.normalization.batch_normalization(net)
		net = tflearn.activations.relu(net)
		net = tflearn.fully_connected(net, 512)
		net = tflearn.layers.normalization.batch_normalization(net)
		net = tflearn.activations.relu(net)
		net = tflearn.fully_connected(net, 512)
		net = tflearn.layers.normalization.batch_normalization(net)
		net = tflearn.activations.relu(net)
		# Final layer weights are init to Uniform[-3e-3, 3e-3]
		w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
		out1 = tflearn.fully_connected(
			net, 1, activation='sigmoid', weights_init=w_init)

		# Scale output to -action_bound to action_bound
		out2 = tflearn.fully_connected(
			net, 1, activation='tanh', weights_init=w_init)
		# out=[out1,out2[]]
		out=tflearn.layers.merge_ops.merge ([out1,out2], 'concat', axis=1, name='Merge')
		scaled_out = tf.multiply(out, self.action_bound)
		return inputs, out, scaled_out

	def train(self, inputs, a_gradient):
		self.sess.run(self.optimize, feed_dict={
			self.inputs: inputs,
			self.action_gradient: a_gradient
		})

	def predict(self, inputs):
		return self.sess.run(self.scaled_out, feed_dict={
			self.inputs: inputs
		})

	def predict_target(self, inputs):
		return self.sess.run(self.target_scaled_out, feed_dict={
			self.target_inputs: inputs
		})

	def update_target_network(self):
		self.sess.run(self.update_target_network_params)

	def get_num_trainable_vars(self):
		return self.num_trainable_vars


class CriticNetwork(object):
	"""
	Input to the network is the state and action, output is Q(s,a).
	The action must be obtained from the output of the Actor network.
	"""

	def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
		self.sess = sess
		self.s_dim = state_dim
		self.a_dim = action_dim
		self.learning_rate = learning_rate
		self.tau = tau
		self.gamma = gamma

		# Create the critic network
		self.inputs, self.action, self.out = self.create_critic_network()

		self.network_params = tf.trainable_variables()[num_actor_vars:]

		# Target Network
		self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

		self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

		# Op for periodically updating target network with online network
		# weights with regularization
		self.update_target_network_params = \
			[self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
			+ tf.multiply(self.target_network_params[i], 1. - self.tau))
				for i in range(len(self.target_network_params))]

		# Network target (y_i)
		self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

		# Define loss and optimization Op
		self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
		self.optimize = tf.train.AdamOptimizer(
			self.learning_rate).minimize(self.loss)

		# Get the gradient of the net w.r.t. the action.
		# For each action in the minibatch (i.e., for each x in xs),
		# this will sum up the gradients of each critic output in the minibatch
		# w.r.t. that action. Each output is independent of all
		# actions except for one.
		self.action_grads = tf.gradients(self.out, self.action)

	def create_critic_network(self):
		inputs = tflearn.input_data(shape=[None, self.s_dim])
		action = tflearn.input_data(shape=[None, self.a_dim])
		net = tflearn.fully_connected(inputs, 512)
		net = tflearn.layers.normalization.batch_normalization(net)
		net = tflearn.activations.relu(net)

		# Add the action tensor in the 2nd hidden layer
		# Use two temp layers to get the corresponding weights and biases
		t1 = tflearn.fully_connected(net, 512)
		t2 = tflearn.fully_connected(action, 512)

		net = tflearn.activation(
			tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

		net = tflearn.fully_connected(net, 512)
		net = tflearn.layers.normalization.batch_normalization(net)
		net = tflearn.activations.relu(net)

		# linear layer connected to 1 output representing Q(s,a)
		# Weights are init to Uniform[-3e-3, 3e-3]


		w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
		out = tflearn.fully_connected(net, 1, weights_init=w_init)
		return inputs, action, out

	def train(self, inputs, action, predicted_q_value):
		return self.sess.run([self.out, self.optimize], feed_dict={
			self.inputs: inputs,
			self.action: action,
			self.predicted_q_value: predicted_q_value
		})

	def predict(self, inputs, action):
		return self.sess.run(self.out, feed_dict={
			self.inputs: inputs,
			self.action: action
		})

	def predict_target(self, inputs, action):
		return self.sess.run(self.target_out, feed_dict={
			self.target_inputs: inputs,
			self.target_action: action
		})

	def action_gradients(self, inputs, actions):
		return self.sess.run(self.action_grads, feed_dict={
			self.inputs: inputs,
			self.action: actions
		})

	def update_target_network(self):
		self.sess.run(self.update_target_network_params)

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
	def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
		self.theta = theta
		self.mu = mu
		self.sigma = sigma
		self.dt = dt
		self.x0 = x0
		self.reset()

	def __call__(self):
		x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
				self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
		self.x_prev = x
		return x

	def reset(self):
		self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

	def __repr__(self):
		return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)




class stateMsg():

	def __init__(self):

		
		self.state = np.zeros(14);
		self.lstate = np.zeros(14);
		self.sub = rospy.Subscriber('/state', Floats, self.fetch)
		self.pub = rospy.Publisher('/cmd_vel_mux/input/navi',Twist,queue_size=10)
		self.pub2 = rospy.Publisher('/resetSignal',Empty,queue_size=10,latch=True)
		self.fpass=1
		self.move_cmd = Twist()
		self.move_cmd.linear.x = 0
		self.move_cmd.angular.z = 0
		self.max_episodes=20000
		self.episode_length=20 #maybe change later
		self.num_episodes=0
		self.terminal=0
		# self.rBuf=ReplayBuffer()
		self.received=0
		with tf.Session() as sess:
			np.random.seed(int(1234))
			tf.set_random_seed(int(1234))
			# env.seed(int(args['random_seed']))
			state_dim = 14
			action_dim = 2
			action_bound =1 #check later
			actor_lr=0.0001
			critic_lr=0.001
			gamma=0.99
			tau=0.01
			minibatch_size=5
			buffer_size=100000
			actor = ActorNetwork(sess, state_dim, action_dim, action_bound,actor_lr,tau,minibatch_size)

			critic = CriticNetwork(sess, state_dim, action_dim,critic_lr,tau,gamma,actor.get_num_trainable_vars())
			
			actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

			self.train(sess,  actor, critic, actor_noise,buffer_size,minibatch_size)

	def fetch(self,msg):
		if self.received==0:
			self.state=np.array(msg.data)
			self.state=np.concatenate((self.state,np.array([self.move_cmd.linear.x,self.move_cmd.angular.z])))
			#put something here for received state
			# print "check"
			self.received=1

	def getstate(self,a):
		self.move_cmd.linear.x = a[0]
		self.move_cmd.angular.z = a[1]
		self.pub.publish(self.move_cmd)
		rospy.sleep(1.)
		self.received=0
		while(self.received==0):
			pass
		R=self.reward()	
		# print R	
		return self.state,R

	def build_summaries(self):
	    episode_reward = tf.Variable(0.)
	    tf.summary.scalar("Reward", episode_reward)
	    episode_ave_max_q = tf.Variable(0.)
	    tf.summary.scalar("Qmax_Value", episode_ave_max_q)

	    summary_vars = [episode_reward, episode_ave_max_q]
	    summary_ops = tf.summary.merge_all()

	    return summary_ops, summary_vars
	
	def train(self,sess, actor, critic, actor_noise,buffer_size,minibatch_size):

	# Set up summary Ops
		summary_ops, summary_vars = self.build_summaries()

		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter("./results", sess.graph)

		# Initialize target network weights
		actor.update_target_network()
		critic.update_target_network()

		# Initialize replay memory
		replay_buffer = ReplayBuffer(int(buffer_size), int(1234))

		for i in range(self.max_episodes):

			# s = env.reset()
			self.pub2.publish()
			# print "reset called"
			ep_reward = 0
			ep_ave_max_q = 0

			for j in range(self.episode_length):
				if j==0:
					# print "first round"
					s,R=self.getstate([0,0])
					R=0
					self.lstate=s
					print R
					continue

				# Added exploration noise
				#a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))

				#********************************wait for state here for the first pass or initis

				

				##*******************************interact here with environment
				#perform action and wait for new state
				# a=np.array([0.1,0])
				a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()
				# print a[0]
				s2,R=self.getstate(a[0])
				print R

				replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), R,
								  self.terminal, np.reshape(s2, (actor.s_dim,)))

				# Keep adding experience to the memory until
				# there are at least minibatch size samples
				if replay_buffer.size() > int(minibatch_size):
					s_batch, a_batch, r_batch, t_batch, s2_batch = \
						replay_buffer.sample_batch(int(minibatch_size))

					# Calculate targets
					target_q = critic.predict_target(
						s2_batch, actor.predict_target(s2_batch))

					y_i = []
					for k in range(int(minibatch_size)):
						if t_batch[k]:
							y_i.append(r_batch[k])
						else:
							y_i.append(r_batch[k] + critic.gamma * target_q[k])

					# Update the critic given the targets
					predicted_q_value, _ = critic.train(
						s_batch, a_batch, np.reshape(y_i, (int(minibatch_size), 1)))

					ep_ave_max_q += np.amax(predicted_q_value)

					# Update the actor policy using the sampled gradient
					a_outs = actor.predict(s_batch)
					grads = critic.action_gradients(s_batch, a_outs)
					actor.train(s_batch, grads[0])

					# Update target networks
					actor.update_target_network()
					critic.update_target_network()

				# s = s2
				s=s2
				self.lstate=s
				ep_reward += R

				if self.terminal==1:
					self.terminal=0
					# print "terminal!!!!!!!!!"
					summary_str = sess.run(summary_ops, feed_dict={
					    summary_vars[0]: ep_reward,
					    summary_vars[1]: ep_ave_max_q / float(j)
					})

					writer.add_summary(summary_str, i)
					writer.flush()

					print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
					        i, (ep_ave_max_q / float(j))))
					break

	def reward(self):
		dist=self.state[10]
		# print dist
		ldist=self.lstate[10]
		if dist<0.2:
			R=10.0
			self.terminal=1
			self.num_episodes+=1
			# print "reached"
		elif dist==1234:
			R=-100.0
			self.terminal=1
			self.num_episodes+=1
			# print "died"
		else:
			R=0.1*(ldist-dist)
			# print "normal"

		return R



def learner():
	rospy.init_node('learner')
	rospy.loginfo("Setting up!")
	stateMsg()
	rospy.spin()

if __name__ == '__main__':
	try:
		learner()
		
		
	except rospy.ROSInterruptException:
		rospy.loginfo("exception")