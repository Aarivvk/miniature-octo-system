import carla
import random
# import time
import math

class EgoHandler:
    
    def __init__(self):
        # init carla client and world
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        # time.sleep(1)
        self.world = client.get_world()
        self.ego_vehicle = self.__get_ego_vehicle()
        self.physics_control = self.ego_vehicle.get_physics_control()        
        # settings = self.world.get_settings()
        # settings.synchronous_mode = True
        # self.world.apply_settings(settings)

    def get_max_steering_angle(self):
        max = 0
        for wheel in self.physics_control.wheels:
            if max < wheel.max_steer_angle:
                max = wheel.max_steer_angle
        max = math.radians(max)
        return max

    def __get_ego_vehicle(self):
        # find actor 'ego_vehicle'
        ego_id = ""
        while ego_id == "":
            # print("waiting to get Vehicle...")
            actors = self.world.get_actors().filter('vehicle.*')
            for actor in actors:
                if(actor.attributes.get('role_name') == 'ego_vehicle'):
                    ego_id = actor.id

        ego_vehicle = self.world.get_actor(ego_id)

        return ego_vehicle
    
    def reset_ego_location(self):
        self.ego_vehicle.set_simulate_physics(False)
        spawn_points = self.world.get_map().get_spawn_points()
        transform = random.choice(spawn_points)
        self.ego_vehicle.set_transform(transform)
        self.ego_vehicle.set_simulate_physics(True)
        # time.sleep(1)
        
    def step(self):
        self.world.tick()
        v = self.ego_vehicle.get_velocity()
        kmph = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        return kmph

    def destory(self):
        pass

