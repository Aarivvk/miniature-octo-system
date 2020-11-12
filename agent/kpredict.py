from tensorflow.keras.models import load_model
import tensorflow as tf
#from tensorflow.keras.utils import print_summary
from carla_gym_env.CarlaGymEnv import CarlaEnv

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

env = CarlaEnv()

actor = load_model("agent/saved/final_carla_target_actor")
print("Loaded the moduel actor")
actor.summary()
 
print("Running the saved model")
# evaluate loaded model on test data
prev_state = env.reset()
done = False
while not done:
   tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
   action = actor.predict(tf_prev_state)
   prev_state, reward, done, info = env.step(action[0])
   # env.render()
env.close()
