import os

import cv2
import numpy as np
from tf_agents.environments import py_environment, utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step

import math
import rospy
from ackermann_msgs.msg import AckermannDrive
from carla_msgs.msg import CarlaCollisionEvent
from carla_msgs.msg import CarlaControl
from carla_msgs.msg import CarlaLaneInvasionEvent

from sensor_msgs.msg import Image

from carla_gym_env.ego import EgoHandler


class CarlaEnv(py_environment.PyEnvironment):

	def __init__(self):
		super().__init__()
		
		rospy.init_node('carla_ego_gym', anonymous=True)
		self.rate = rospy.Rate(5) # 100Hz
		
		self.pub_ackermann_cmd = rospy.Publisher('/carla/ego_vehicle/ackermann_cmd', AckermannDrive, queue_size=5)
		self.pub_carla_control_cmd = rospy.Publisher('/carla/control', CarlaControl, queue_size=5)
		
		self.image_subscriber = rospy.Subscriber("/carla/ego_vehicle/camera/rgb/front/image_color", Image, self.on_image)
		self.collision_subscriber = rospy.Subscriber("/carla/ego_vehicle/collision", CarlaCollisionEvent, self.on_collision)
		self.lane_invasion_subscriber = rospy.Subscriber("/carla/ego_vehicle/lane_invasion", CarlaLaneInvasionEvent, self.on_lane_invasion)
		
		self.ego = EgoHandler()

		# sensors data
		self.data_cam_front_rgb = np.zeros((84, 84, 3), dtype=np.float32)
		self._reset_data()
		# control data
		self.carla_ctrl_cmd = CarlaControl()
		self.carla_ctrl_cmd.command = self.carla_ctrl_cmd.STEP_ONCE
		self.ackermann_cmd = AckermannDrive()

		# History data
		self.last_steering = 0.0

		self._action_spec = array_spec.BoundedArraySpec(shape=(2,), dtype=np.float32, minimum=[0.0, -0.5], maximum=[5.0, 0.5], name='action')
		self._observation_spec = array_spec.BoundedArraySpec(shape=(84, 84, 3), dtype=np.float32, minimum=0, maximum=1, name='observation')

		self._send_command([0,0])


	def action_spec(self):
		return self._action_spec


	def observation_spec(self):
		return self._observation_spec


	def _send_command(self, action):
		# form action msg format
		self.ackermann_cmd.speed = action[0] # speed
		self.ackermann_cmd.steering_angle = action[1] # steering_angle
		# execute action over ros
		self.pub_ackermann_cmd.publish(self.ackermann_cmd)
		# step-in command for sycronus Enable in ros-bridge.
		self.pub_carla_control_cmd.publish(self.carla_ctrl_cmd)


	def _reset(self):
		# print(f"Resetting environment collid: {self.is_ego_collided} lane_crossed:{self.data_lane_crossed}")
		self._reset_data()
		self.ego.reset_ego_location()
		self._send_command([0,0])
		self.rate.sleep()
		return time_step.restart(self.data_cam_front_rgb)


	def _step(self, action):
		if self.isEgoViolatedTraffic():
			# The last action ended the episode. Ignore the current action and start a new episode.
			return self._reset()

		self._send_command(action)
		if action[0] == 0.0:
			reward = -1
		else:
			reward = action[0]/4.5 - abs(self.last_steering - action[1])/self._action_spec.maximum[1]
			self.last_steering = action[1]
		# needed for ros msg to sync.
		self.rate.sleep()
		# return transition depending on game state
		if self.isEgoViolatedTraffic():
			reward = -1
			return time_step.termination(self.data_cam_front_rgb, reward)
		else:
			return time_step.transition(self.data_cam_front_rgb, reward)

	def _reset_data(self):
		self.last_steering = 0.0
		self.data_collision_intensity = 0.0
		self.data_lane_crossed = False
		self.is_ego_collided = False


	def render(self, mode='rgb_array'):
		""" Return image for rendering. """
		# cv2.imshow("vk", self.data_cam_front_rgb)
		# cv2.waitKey(1)
		return self.data_cam_front_rgb

	def isEgoViolatedTraffic(self):
		game_end = (self.data_lane_crossed or self.is_ego_collided)
		if game_end:
			self.data_lane_crossed = False
		return game_end
		

	def on_lane_invasion(self, data):
		"""
		Callback on lane invasion event
		"""
		text = []
		for marking in data.crossed_lane_markings:
			if marking is CarlaLaneInvasionEvent.LANE_MARKING_OTHER:
				text.append("Other")
			elif marking is CarlaLaneInvasionEvent.LANE_MARKING_BROKEN:
				text.append("Broken")
			elif marking is CarlaLaneInvasionEvent.LANE_MARKING_SOLID:
				text.append("Solid")
				self.data_lane_crossed = True
			else:
				text.append("Unknown ")

	def on_image(self, image):
		"""
		Callback when receiving a camera image
		"""
		array = np.frombuffer(image.data, dtype=np.uint8)
		array = np.reshape(array, (image.height, image.width, 4))
		array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
		array = np.divide(array, 255, dtype=np.float32)
		array = cv2.resize(array, (84, 84))
		# array = array[-42:,:]
		self.data_cam_front_rgb = array

	def on_collision(self, data):
		"""
		Callback on collision event
		"""
		self.data_collision_intensity = math.sqrt(data.normal_impulse.x**2 +
								data.normal_impulse.y**2 + data.normal_impulse.z**2)
		if self.data_collision_intensity > 0.0:
			self.is_ego_collided = True

if __name__ == "__main__":
	environment = CarlaEnv()
	utils.validate_py_environment(environment, episodes=5)
