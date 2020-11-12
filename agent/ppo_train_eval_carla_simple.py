from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import copy

import tensorflow as tf
from absl import flags, app
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies.policy_saver import PolicySaver
import imageio

from signal import signal, SIGINT

import threading

from carla_gym_env.CarlaEnv import CarlaEnv
# from utils.visualization_helper import create_video

flags.DEFINE_string('videos_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'), 'Directory to write evaluation videos to')
FLAGS = flags.FLAGS

terminate = False
counter = 0

def create_video(tf_environment, policy, num_episodes=10, video_filename='imageio.mp4'):
	print("Generating video %s" % video_filename)
	with imageio.get_writer(video_filename, fps=60) as video:
		for episode in range(num_episodes):
			print("Generating episode %d of %d" % (episode, num_episodes))

			time_step = tf_environment.reset()
			state = policy.get_initial_state(tf_environment.batch_size)

			video.append_data(tf_environment.render())
			while not time_step.is_last():
				PolicyStep = policy.action(time_step, state)
				state = PolicyStep.state
				time_step = tf_environment.step(PolicyStep.action)
				video.append_data(tf_environment.render())

	print("Finished video %s" % video_filename)

def handler(signal_received, frame):
    # Handle any cleanup here
	print('SIGINT or CTRL-C detected. Exiting gracefully')
	global terminate 
	global counter
	terminate = True
	counter = counter + 1
	if(counter == 3):
		exit(0)

def create_networks(observation_spec, action_spec):
	actor_net = ActorDistributionRnnNetwork(
		observation_spec,
		action_spec,
		conv_layer_params=[(64, 8, 4), (32, 4, 2), (16, 4, 2)],
		input_fc_layer_params=(256,),
		lstm_size=(256,),
		output_fc_layer_params=(128,),
		activation_fn=tf.keras.activations.relu)
	value_net = ValueRnnNetwork(
		observation_spec,
		conv_layer_params=[(64, 8, 4), (32, 4, 2), (16, 4, 2)],
		input_fc_layer_params=(256,),
		lstm_size=(256,),
		output_fc_layer_params=(128,),
		activation_fn=tf.keras.activations.relu)

	return actor_net, value_net


def train_eval_doom_simple(
		# Params for collect
		num_environment_steps=100000,
		collect_episodes_per_iteration=32,
		num_parallel_environments=1,
		replay_buffer_capacity=301,  # Per-environment
		# Params for train
		num_epochs=25,
		learning_rate=4e-4,
		# Params for eval
		eval_interval=10,
		num_video_episodes=10,
		# Params for summaries and logging
		log_interval=10):
	"""A simple train and eval for PPO."""
	# if not os.path.exists(videos_dir):
	# 	os.makedirs(videos_dir)
	global terminate
	eval_py_env = CarlaEnv()
	tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

	actor_net, value_net = create_networks(tf_env.observation_spec(), tf_env.action_spec())

	global_step = tf.compat.v1.train.get_or_create_global_step()
	optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)

	tf_agent = ppo_agent.PPOAgent(
		tf_env.time_step_spec(),
		tf_env.action_spec(),
		optimizer,
		actor_net,
		value_net,
		num_epochs=num_epochs,
		train_step_counter=global_step,
		discount_factor=0.99,
		gradient_clipping=0.5,
		entropy_regularization=1e-2,
		importance_ratio_clipping=0.2,
		use_gae=True,
		use_td_lambda_return=True
	)
	tf_agent.initialize()

	environment_steps_metric = tf_metrics.EnvironmentSteps()
	step_metrics = [
		tf_metrics.NumberOfEpisodes(),
		environment_steps_metric,
	]

	replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(tf_agent.collect_data_spec, batch_size=num_parallel_environments, max_length=replay_buffer_capacity)
	train_replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(tf_agent.collect_data_spec, batch_size=num_parallel_environments, max_length=replay_buffer_capacity)
	collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(tf_env, tf_agent.collect_policy, observers=[replay_buffer.add_batch] + step_metrics, num_episodes=collect_episodes_per_iteration)

	collect_time = 0
	train_time = 0
	timed_at_step = global_step.numpy()
	
	my_policy = tf_agent.policy
	saver = PolicySaver(my_policy, batch_size=None)

	def train_step():
		trajectories = train_replay_buffer.gather_all()
		return tf_agent.train(experience=trajectories)
	
	def evaluate(policy, step_count):
		create_video(tf_env, policy, 10, f'agent/behave/imageio_{step_count}.mp4')


	print("collecting samples initial:")
	collect_driver.run()
	train_replay_buffer = copy.deepcopy(replay_buffer)
	replay_buffer.clear()
	print(f"train size {train_replay_buffer.num_frames()} buffer size{replay_buffer.num_frames()}")

	while environment_steps_metric.result() < num_environment_steps and not terminate:
		start_time = time.time()
		print("collecting samples")
		collector_thread = threading.Thread(target=collect_driver.run)
		collector_thread.start()

		start_time = time.time()
		count = 0
		# while collector_thread.is_alive() and not terminate:
		# 	count = count + 1
		print(f"Training agent {count}")
		total_loss, _ = train_step()
		print()
		print("'''''''''''''''''''''''''''''''''''Tensorflow logs:'''''''''''''''''''''''''''''''''''")
		print(f'step = {global_step.numpy()}, loss = {total_loss}, env_metric = {environment_steps_metric.result()}')
		print("'''''''''''''''''''''''''''''''''''Tensorflow logs:'''''''''''''''''''''''''''''''''''")
		print()
		train_replay_buffer.clear()
		print("Training agent Finshed")
		print("Waiting for collecting samples thread")
		collector_thread.join()
		print("collecting samples Finished")
		collect_time += time.time() - start_time
		train_replay_buffer = copy.deepcopy(replay_buffer)
		replay_buffer.clear()
		train_time += time.time() - start_time

		global_step_val = global_step.numpy()

		print(f"global_step_val:{global_step_val} % log_interval:{log_interval} = {global_step_val % log_interval}")

		# if global_step_val % log_interval == 0:
		print()
		print("'''''''''''''''''''''''''''''''''''Tensorflow logs:'''''''''''''''''''''''''''''''''''")
		print(f'step = {global_step_val}, loss = {total_loss}')
		steps_per_sec = ((global_step_val - timed_at_step) / (collect_time + train_time))
		print(f'{steps_per_sec} steps/sec')
		print(f'collect_time = {collect_time}, train_time = {train_time}')
		print("'''''''''''''''''''''''''''''''''''Tensorflow logs:'''''''''''''''''''''''''''''''''''")
		print()
		timed_at_step = global_step_val
		collect_time = 0
		train_time = 0

		if global_step_val % eval_interval == 0:
			print("Evaluating!!")
			saver.save(f'agent/saved/policy_ppo_simple_{global_step_val}')
			policy = tf_agent.policy
			evaluate(policy, global_step_val)

	print("Terminated")
	policy = tf_agent.policy
	evaluate(policy, global_step_val)


def main(_):
	signal(SIGINT, handler)
	physical_devices = tf.config.list_physical_devices('GPU') 
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
	# tf.compat.v1.enable_v2_behavior()  # For TF 1.x users
	train_eval_doom_simple()


if __name__ == '__main__':
	# flags.mark_flag_as_required('videos_dir')
	app.run(main)
