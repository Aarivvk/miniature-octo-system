from tensorflow.keras.models import load_model
import tensorflow as tf
#from tensorflow.keras.utils import print_summary
import gym
import gym_carla

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def createEnv():
  # parameters for the gym_carla environment
  params = {
    'number_of_vehicles': 0,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.02,  # time interval between two frames
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

actor = load_model("agent/saved/final_carla_target_actor")
print("Loaded the moduel actor")
actor.summary()
 
print("Running the saved model")
# evaluate loaded model on test data
for i in range(10):
   done = False
   prev_state = env.reset()['birdeye']
   print(f"running episode {i}")
   while not done:
      tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
      action = actor.predict(tf_prev_state)
      prev_state, reward, done, info = env.step(action[0])
      prev_state = prev_state['birdeye']
      # env.render()
env.close()
