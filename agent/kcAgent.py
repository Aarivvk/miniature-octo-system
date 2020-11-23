#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import numpy as np
import matplotlib.pyplot as plt

import gym
import gym_carla

import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import plot_model

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

plt.xlabel("Episode")
plt.ylabel("Epsiodic Reward")
plt.title('Rewards look-up')

def plot(ep, reward, al, cl):
    plt.scatter(ep, reward, color="black")
    plt.scatter(ep, al, color="red")
    plt.scatter(ep, cl, color="green")
    plt.draw()
    plt.pause(0.0005)

plot(0, 0, 0, 0)

def createEnv():
  # parameters for the gym_carla environment
  params = {
    'number_of_vehicles': 0,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': False,  # whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 2000,  # connection port
    'town': 'Town03',  # which town to simulate
    'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 1000,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'obs_rsize': 256, # resize value.
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
  }

  # Set gym-carla environment
  env = gym.make('carla-v0', params=params)

  return env

env = createEnv()

num_states = env.observation_space['birdeye'].shape
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape
print("Size of Action Space ->  {}".format(num_actions))

speed_upper_bound = env.action_space.high[0]
speed_lower_bound = env.action_space.low[0]
print("Max Value of Speed Action ->  {}".format(speed_upper_bound))
print("Min Value of Speed Action ->  {}".format(speed_lower_bound))

steering_upper_bound = env.action_space.high[1]
steering_lower_bound = env.action_space.low[1]
print("Max Value of Steering Action ->  {}".format(steering_upper_bound))
print("Min Value of Steering Action ->  {}".format(steering_lower_bound))

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, *num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, *num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, *num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )
        return (actor_loss, critic_loss)

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        return self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape = num_states)
    out = layers.Conv2D(24, kernel_size=(20, 20), strides=2, activation='relu')(inputs)
    out = layers.MaxPooling2D(pool_size = (7, 7))(out)
    out = layers.Conv2D(48, kernel_size=(15, 15), strides=2, activation='relu')(out)
    out = layers.MaxPooling2D(pool_size = (4, 4), padding="same")(out)
    out = layers.Conv2D(96, kernel_size=(10, 10), padding="same", strides=2, activation='relu')(out)
    out = layers.Conv2D(5, kernel_size=(1, 1), strides=2, activation='softmax')(out)
    out = layers.Flatten()(out)
    # out = layers.Dense(256, activation="relu")(out)
    out = layers.Dense(128, activation="relu")(out)
    out = layers.Dense(64, activation="relu")(out)
    out = layers.Dense(32, activation="relu")(out)
    out = layers.Dense(2, activation="tanh", kernel_initializer=last_init)(out)

    model = tf.keras.Model(inputs, out)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape = num_states)
    state_out = layers.Conv2D(24, kernel_size=(20, 20), strides=2, activation='relu')(state_input)
    state_out = layers.MaxPooling2D(pool_size = (7, 7))(state_out)
    state_out = layers.Conv2D(48, kernel_size=(15, 15), strides=2, activation='relu')(state_out)
    state_out = layers.MaxPooling2D(pool_size = (4, 4), padding="same")(state_out)
    state_out = layers.Conv2D(96, kernel_size=(10, 10), padding="same", strides=2, activation='relu')(state_out)
    state_out = layers.Conv2D(5, kernel_size=(1, 1), strides=2, activation='softmax')(state_out)
    state_out = layers.Flatten()(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(64, activation="relu")(concat)
    out = layers.Dense(32, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    # sampled_actions = tf.squeeze(actor_model.predict(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    # legal_action = np.zeros((*num_actions,), dtype=float)
    sampled_actions = np.clip(sampled_actions, -1, 1)
    if (sampled_actions[0] or sampled_actions[1]) < -1 or (sampled_actions[0] or sampled_actions[1]) > 1:
        print(f"Problem {sampled_actions}")

    return np.squeeze(sampled_actions)

std_dev = 0.1
ou_noise = OUActionNoise(mean=np.zeros(2), std_deviation=float(std_dev) * np.ones(2))



actor_model = get_actor()
actor_model.summary()
critic_model = get_critic()
actor_model.summary()

# plot_model(actor_model, to_file='actor_model.png')
# plot_model(critic_model, to_file='critic_model.png')

if os.path.isdir("agent/saved/final_carla_target_actor"):
    print("Loading the previously stored network")
    actor_loaded = load_model("agent/saved/final_carla_target_actor")
    critic_loaded = load_model("agent/saved/final_carla_target_critic")
    actor_model.set_weights(actor_loaded.get_weights())
    critic_model.set_weights(critic_loaded.get_weights())
    print("Loading is complete!!")


target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

actor_loss = None
critic_loss = None

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 500
max_num_per_epi = 250
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.001

buffer = Buffer(5000, 16)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
# To track when to save the network.
saved_interval = 50

input_source = 'birdeye' # 'birdeye', # 'camera', # 'lidar'

episodic_reward = 0
# Takes about 4 min to train
for ep in range(total_episodes):
    prev_state = env.reset()[input_source]
    episodic_reward = 0

    # for ep_step in range(max_num_per_epi):
    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        # env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy(tf_prev_state, ou_noise)
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)

        buffer.record((prev_state, action, reward, state[input_source]))
        episodic_reward += reward

        actor_loss, critic_loss = buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        # End this episode when `done` is True
        if done:
            # prev_state = env.reset()[input_source]
            # done = False
            break
        else:
            prev_state = state[input_source]

    # Plotting graph
    plot(ep, episodic_reward, actor_loss, critic_loss)

    ep_reward_list.append(episodic_reward)

    # Mean of last 50 episodes
    avg_reward = np.mean(ep_reward_list[-50:])
    avg_reward_list.append(avg_reward)
    
    str1 = f"| Episode * {ep} * Avg Reward is ==> {avg_reward:.2f} * Ep. Reward {episodic_reward:.2f}"
    str2 = f"| Actor loss ==> {actor_loss:.2f} * Critic loss ==> {critic_loss:.2f}"
    str_lenth = max(len(str1), len(str2))
    print("-"*str_lenth)
    print(str1 + (" " * (str_lenth - len(str1))) + "|")
    print(str2+ (" " * (str_lenth - len(str2))) + "|")
    print("-"*str_lenth)
    print()

    if(ep != 0 and (ep % saved_interval) == 0):
        print(f"Saved the target actor and critic model Epi num.:{ep}")
        # Save the weights
        target_actor.save(f"agent/saved/carla_target_actor_{ep}")
        target_critic.save(f"agent/saved/carla_target_critic_{ep}")


print("Saved the target actor model")
# Save the weights
target_actor.save("agent/saved/final_carla_target_actor")
target_critic.save("agent/saved/final_carla_target_critic")

plt.show()
print("Terminated !")