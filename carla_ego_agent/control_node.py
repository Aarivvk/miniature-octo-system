import math
import argparse
import rospy
from ackermann_msgs.msg import AckermannDrive
from carla_msgs.msg import CarlaCollisionEvent
from carla_msgs.msg import CarlaControl

from ego import EgoHandler

from signal import signal, SIGINT

recived_data = None
terminate = False

def handler(signal_received, frame):
    # Handle any cleanup here
    global terminate
    terminate = True

def calculate_impulse(data):
     intensity = math.sqrt(data.normal_impulse.x**2 + data.normal_impulse.y**2 + data.normal_impulse.z**2)
     return intensity

def callback(data):
    # print(f"Recived the data /carla/ego_vehicle/collision x:{data.normal_impulse.x}, y:{data.normal_impulse.y}, z:{data.normal_impulse.z}")
    global recived_data
    recived_data = data

def control_vehicle(speed, steering_angle):
    global recived_data
    rospy.init_node('vk_vehicle_controller', anonymous=True)
    pub_ackermann_cmd = rospy.Publisher('/carla/ego_vehicle/ackermann_cmd', AckermannDrive, queue_size=5)
    pub_carla_control_cmd = rospy.Publisher('/carla/control', CarlaControl, queue_size=5)
    subscriber = rospy.Subscriber('/carla/ego_vehicle/collision', CarlaCollisionEvent, callback)
    rate = rospy.Rate(10)
    ego = EgoHandler()
    while not rospy.is_shutdown() and not terminate:
        carla_ctrl_cmd = CarlaControl()
        ackermann_cmd = AckermannDrive()

        if recived_data is not None and (calculate_impulse(recived_data) > 10):
            print(f"Colission detected impact :{recived_data.normal_impulse.x}")
            recived_data.normal_impulse.x = 0
            ego.reset_ego_location()
        else:
            ackermann_cmd.speed = speed+30
            ackermann_cmd.steering_angle = steering_angle

        carla_ctrl_cmd.command = carla_ctrl_cmd.STEP_ONCE
        pub_ackermann_cmd.publish(ackermann_cmd)
        pub_carla_control_cmd.publish(carla_ctrl_cmd)
        print(f"Step in to {carla_ctrl_cmd.STEP_ONCE}")
    print("Program terminated")

if __name__ == '__main__':

    signal(SIGINT, handler)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--steering_angle", default = 0.0, help="steering angle input (rad)")
    parser.add_argument("--speed", default = 5.0, help="speed input (m/s)")
    args = parser.parse_args()

    try:
        control_vehicle(args.speed, args.steering_angle)
    except rospy.ROSInterruptException:
        pass
