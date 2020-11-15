import os

import cv2
import numpy as np
from numpy import asarray

import gym
from gym import error, spaces, utils
from gym.utils import seeding

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


class CarlaEnv(gym.Env):

    def __init__(self):
        super().__init__()
        metadata = {'render.modes': ['human']}

        rospy.init_node('carla_ego_gym', anonymous=True)
        self.rate = rospy.Rate(20)  # 100Hz

        # control data
        self.pub_ackermann_cmd = rospy.Publisher(
            '/carla/ego_vehicle/ackermann_cmd', AckermannDrive, queue_size=5)
        self.ackermann_cmd = AckermannDrive()
        self.pub_carla_control_cmd = rospy.Publisher(
            '/carla/control', CarlaControl, queue_size=5)
        self.carla_ctrl_cmd = CarlaControl()
        self.carla_ctrl_cmd.command = self.carla_ctrl_cmd.PLAY
        self.pub_carla_control_cmd.publish(self.carla_ctrl_cmd)
        self.carla_ctrl_cmd.command = self.carla_ctrl_cmd.STEP_ONCE

        self.image_subscriber = rospy.Subscriber(
            "/carla/ego_vehicle/camera/rgb/front/image_color", Image, self.on_image)
        self.collision_subscriber = rospy.Subscriber(
            "/carla/ego_vehicle/collision", CarlaCollisionEvent, self.on_collision)
        self.lane_invasion_subscriber = rospy.Subscriber(
            "/carla/ego_vehicle/lane_invasion", CarlaLaneInvasionEvent, self.on_lane_invasion)

        self.ego = EgoHandler()

        # sensors data
        self._reset_data()
        self._steering_threshold = 0.3

        # History data
        self.last_steering = 0.0
        max_steering = self.ego.get_max_steering_angle()
        self.total_reward = 0.0

        self._action_spec = array_spec.BoundedArraySpec(shape=(2,), dtype=np.float32, minimum=[
                                                        0.0, (-1 * max_steering)], maximum=[12.0, max_steering], name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(84, 84, 3), dtype=np.float32, minimum=0, maximum=1, name='observation')

        self._send_command([0, 0])

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _send_command(self, action):
        # form action msg format
        self.ackermann_cmd.speed = action[0]  # speed
        self.ackermann_cmd.steering_angle = action[1]  # steering_angle
        # execute action over ros
        self.pub_ackermann_cmd.publish(self.ackermann_cmd)
        # step-in command for sycronus Enable in ros-bridge.
        self.pub_carla_control_cmd.publish(self.carla_ctrl_cmd)

    def reset(self):
        # print(f"Resetting environment collid: {self.is_ego_collided} lane_crossed:{self.data_lane_crossed} Total reward {self.total_reward}")
        self.rate.sleep()
        self._reset_data()
        self.ego.reset_ego_location()
        self._send_command([0, 0])
        self.ego.step()
        self.total_reward = 0.0
        return self.data_cam_front_rgb

    def step(self, action):
        # scale back to original values.
        original = action

        # if original[1] < -self._steering_threshold:
        # 	self.left_actions = self.left_actions + 1
        # if original[1] > self._steering_threshold:
        # 	self.right_actions = self.right_actions + 1

        action[0] = (action[0] + 1) * self._action_spec.maximum[0]
        action[1] = action[1] * 0.5 #* self._action_spec.maximum[1]
        self._send_command(action)

        # needed for ros msg to sync.
        self.rate.sleep()
        kmph, reward = self.ego.step()

        done = self.isEgoViolatedTraffic()

        if done:
            # Punish for violating the traffic rules.
            reward = -250
        elif kmph < 35:
            reward = reward - 1 # - abs(original[1])
        else:
            # Reward if accelaration is +ve and punish if -ve.
            reward = 1 + reward # original[0]
            # Punish for jerk and for extreem truns.
            # reward = reward - (abs(original[1]) * 1.5)#- abs(self.last_steering - original[1])
            # As it is for every step and for cumulative reward scale it down.
            # if reward > 0:
            # 	reward = reward / 100.0
            self.last_steering = original[1]
        
        # if (self.right_actions or self.left_actions) >= 40 and not done:
        # 	self.left_actions = 0.0
        # 	self.right_actions = 0.0
        # 	done = True

        # self.total_reward = self.total_reward + reward
        # reward = reward / 10.0
        # return transition depending on game state
        return (self.data_cam_front_rgb, reward, done, "VK")

    def _reset_data(self):
        self._render_img = np.zeros((84, 84, 3), dtype=np.float32)
        self.data_cam_front_rgb = np.zeros((84, 84, 3), dtype=np.float32)
        self.last_steering = 0.0
        self.data_collision_intensity = 0.0
        self.data_lane_crossed = False
        self.is_ego_collided = False
        self.left_actions = 0.0
        self.right_actions = 0.0

    def render(self, mode='human'):
        """ Return image for rendering. """
        # cv2.imshow("vk", self.data_cam_front_rgb)
        # cv2.waitKey(1)
        return asarray(self._render_img).reshape(self._render_img.shape)

    def isEgoViolatedTraffic(self):
        game_end = (self.data_lane_crossed or self.is_ego_collided)
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
                # self.data_lane_crossed = True
                break
            else:
                text.append("Unknown ")

    def on_image(self, image):
        """
        Callback when receiving a camera image
        """
        array = np.frombuffer(image.data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
        self._render_img = array
        # Normalise the array.
        array = np.divide(array, 255, dtype=np.float32)
        array = cv2.resize(array, (84, 84))
        # array = array[-42:,:]
        # array = asarray(array)
        self.data_cam_front_rgb = array

    def on_collision(self, data):
        """
        Callback on collision event
        """
        self.data_collision_intensity = math.sqrt(data.normal_impulse.x**2 +
                                                  data.normal_impulse.y**2 + data.normal_impulse.z**2)
        if self.data_collision_intensity > 0.0:
            self.is_ego_collided = True
        else:
            self.is_ego_collided = False
        # print(f"Collision detected !! {self.data_collision_intensity}")


if __name__ == "__main__":
    environment = CarlaEnv()
    environment.reset()
    done = False
    while not done:
        state, reward, done, info = environment.step([0.0, 0.0])
