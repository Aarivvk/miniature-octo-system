import tensorflow as tf
from carla_gym_env.CarlaEnv import CarlaEnv
from tf_agents.environments import tf_py_environment
import imageio

from numpy import asarray

eval_py_env = CarlaEnv()


def create_video(tf_environment, num_episodes=1, video_filename='imageio.mp4'):
    print("Generating video %s" % video_filename)
    with imageio.get_writer(video_filename, fps=60) as video:
        for episode in range(num_episodes):
            print("Generating episode %d of %d" % (episode, num_episodes))
            time_step = tf_environment.reset()
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print(tf_environment.render())
            print(type(asarray(tf_environment.render()).reshape(84,84,3)))
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            video.append_data(asarray(tf_environment.render()).reshape(84,84,3))
            while not tf_environment.is_last():
                time_step = tf_environment.step([4.0, 0.0])
                video.append_data(asarray(tf_environment.render()))

    print("Finished video %s" % video_filename)

print("Evaluating!!")

create_video(eval_py_env)
