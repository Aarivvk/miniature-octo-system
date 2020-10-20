import carla
import random

class EgoHandler:
    
    def __init__(self):
        # init carla client and world
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        self.world = client.get_world()
        self.ego_vehicle = None
        self.ego_vehicle = self.get_ego_vehicle()

    def get_ego_vehicle(self):
        if self.ego_vehicle is None:
            # find actor 'ego_vehicle'
            actors = self.world.get_actors().filter('vehicle.*')
            for actor in actors:
                if(actor.attributes.get('role_name') == 'ego_vehicle'):
                    ego_id = actor.id
            self.ego_vehicle = self.world.get_actor(ego_id)

        return self.ego_vehicle
    
    def reset_ego_location(self):
        spawn_points = self.world.get_map().get_spawn_points()
        transform = random.choice(spawn_points)
        self.ego_vehicle.set_transform(transform)
        


    def destory():
        pass

