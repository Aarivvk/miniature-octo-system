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
        self.is_auto_enabled = False
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

    def setAutoP(self, value):
        if not self.is_auto_enabled == value:
            self.ego_vehicle.set_autopilot(value)
            self.is_auto_enabled = value

    def convertRange(self, OldValue):
        OldMax = 1
        OldMin = 0
        NewMax = 1
        NewMin = -1
        OldRange = (OldMax - OldMin)
        if (OldRange == 0):
            NewValue = NewMin
        else:
            NewRange = (NewMax - NewMin)  
            NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
        return NewValue

        
    def step(self):
        # life_span = 0.0
        # way_point_p = self.world.get_map().get_waypoint(self.ego_vehicle.get_location())
        # self.world.debug.draw_string(way_point_p.transform.location, 'O', draw_shadow=False,
        #                            color=carla.Color(r=0, g=255, b=0), life_time=life_span,
        #                            persistent_lines=True)
        self.world.tick()
        way_point_n = self.world.get_map().get_waypoint(self.ego_vehicle.get_location())
        # self.world.debug.draw_string(way_point_n.transform.location, 'O', draw_shadow=False,
        #                            color=carla.Color(r=255, g=0, b=0), life_time=life_span,
        #                            persistent_lines=True)
        # expected_wp = way_point_n.next(2.0)[0]
        # print(way_point_n.transform.location)
        off_location = way_point_n.transform.location - self.ego_vehicle.get_location()
        off_location = abs(off_location.x) + abs(off_location.z) + abs(off_location.z)
        off_location = -off_location
        # self.world.debug.draw_string(expected_wp.transform.location, 'X', draw_shadow=False,
        #                            color=carla.Color(r=255, g=0, b=0), life_time=life_span,
        #                            persistent_lines=True)
        v = self.ego_vehicle.get_velocity()
        kmph = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        if self.is_auto_enabled:
            ctrl = self.ego_vehicle.get_control()
            action = [self.convertRange(ctrl.throttle), ctrl.steer]
        else:
            action = [0.0, 0.0]

        return (kmph, off_location, action)

    def destory(self):
        pass

