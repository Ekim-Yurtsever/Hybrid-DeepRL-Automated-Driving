import settings
from sources import operating_system, STOP
import glob
import os
import sys
try:
    sys.path.append(glob.glob(settings.CARLA_PATH + f'/PythonAPI/carla/dist/carla-*{sys.version_info.major}.{sys.version_info.minor}-{"win-amd64" if os.name == "nt" else "linux-x86_64"}.egg')[0])
except IndexError:
    pass

# try:
#     sys.path.append(glob.glob(settings.CARLA_PATH + '/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

import carla
import time
import random
import numpy as np
import math
from dataclasses import dataclass
import psutil
import subprocess
from queue import Queue

# ==============================================================================
# -- Navigation imports ----------------------------------------------------------
# ==============================================================================

from sources.navigation.global_route_planner import GlobalRoutePlanner
from sources.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from sources.navigation.modified_local_planner import ModifiedLocalPlanner

from numpy import random

# ==============================================================================
# -- Get colors for debugging --------------------------------------------------
# ==============================================================================

red = carla.Color(255, 0, 0)
green = carla.Color(0, 255, 0)
blue = carla.Color(47, 210, 231)
cyan = carla.Color(0, 255, 255)
yellow = carla.Color(255, 255, 0)
orange = carla.Color(255, 162, 0)
white = carla.Color(255, 255, 255)

# ==============================================================================


@dataclass
class ACTIONS:
    forward_slow = 0
    forward_medium = 1
    forward_fast = 2
    left_slow = 3
    left_medium = 4
    left_fast = 5
    right_slow = 6
    right_medium = 7
    right_fast = 8
    brake_light = 9
    brake_medium = 10
    brake_full = 11
    no_action = 12

ACTION_CONTROL = {
    0: [0.3, 0, 0],
    1: [0.6, 0, 0],
    2: [1, 0, 0],
    3: [0.7, 0, -0.3],
    4: [0.7, 0, -0.6],
    5: [0.7, 0, -1],
    6: [0.7, 0, 0.3],
    7: [0.7, 0, 0.6],
    8: [0.7, 0, 1],
    9: [0, 0.3, 0],
    10: [0, 0.6, 0],
    11: [0, 1, 0],
    12: None,
}

ACTIONS_NAMES = {
    0: 'forward_slow',
    1: 'forward_medium',
    2: 'forward_fast',
    3: 'left_slow',
    4: 'left_medium',
    5: 'left_fast',
    6: 'right_slow',
    7: 'right_medium',
    8: 'right_fast',
    9: 'brake_light',
    10: 'brake_medium',
    11: 'brake_full',
    12: 'no_action',
}

# Carla environment
class CarlaEnv:

    # How much steering to apply
    STEER_AMT = 1.0

    # Image dimensions (observation space)
    im_width = settings.IMG_WIDTH
    im_height = settings.IMG_HEIGHT

    # Action space size
    action_space_size = len(settings.ACTIONS)

    # ==============================================================================
    # -- Navigation functions -------------------------------------------------------
    # ==============================================================================

    def total_distance(self, current_plan):
        sum = 0
        for i in range(len(current_plan) - 1):
            sum = sum + self.distance_wp(current_plan[i + 1][0], current_plan[i][0])
        return sum

    def distance_wp(self, target, current):
        dx = target.transform.location.x - current.transform.location.x
        dy = target.transform.location.y - current.transform.location.y
        return math.sqrt(dx * dx + dy * dy)

    def distance_goal(self, target, current):
        dx = target.location.x - current.x
        dy = target.location.y - current.y
        return math.sqrt(dx * dx + dy * dy)

    def distance_vehicle(self, waypoint, vehicle_transform):
        loc = vehicle_transform.location
        dx = waypoint.transform.location.x - loc.x
        dy = waypoint.transform.location.y - loc.y

        return math.sqrt(dx * dx + dy * dy)

    def draw_path(self, world, current_plan):
        for i in range(len(current_plan) - 1):
            w1 = current_plan[i][0]
            w2 = current_plan[i + 1][0]
            self.world.debug.draw_line(w1.transform.location, w2.transform.location, thickness=0.5,
                                       color=green, life_time=30.0)
        self.draw_waypoint_info(world, current_plan[-1][0])

    def draw_waypoint_info(self, world, w, lt=30):
        w_loc = w.transform.location
        world.debug.draw_point(w_loc, 0.5, red, lt)

    # ==============================================================================

    def __init__(self, carla_instance, seconds_per_episode=None, playing=False):

        # Set a client and timeouts
        self.client = carla.Client(*settings.CARLA_HOSTS[carla_instance][:2])
        self.client.set_timeout(2.0)

        # Get currently running world
        self.world = self.client.get_world()

        # Get list of actor blueprints
        blueprint_library = self.world.get_blueprint_library()

        # Get Tesla model 3 blueprint
        # self.model_3 = blueprint_library.filter('model3')[0]

        # Get a random blueprint.
        vehicles = self.world.get_blueprint_library().filter("vehicle.*")
        choices = [x for x in vehicles if int(x.get_attribute('number_of_wheels')) == 4]
        self.mycar = random.choice(choices)
        self.obs = random.choice(choices)

        # Sensors and helper lists
        self.collision_hist = []
        self.actor_list = []
        self.front_camera = None
        self.preview_camera = None

        # Used to determine if Carla simulator is still working as it crashes quite often
        self.last_cam_update = time.time()

        # Updated by agents for statistics
        self.seconds_per_episode = seconds_per_episode

        # A flag indicating that we are just going to play
        self.playing = playing

        # Used with additional preview feature
        self.preview_camera_enabled = False

        # Sets actually configured actions
        self.actions = [getattr(ACTIONS, action) for action in settings.ACTIONS]

        # Set large initial distance to goal
        self.prev_d2goal = 10000
        # self.d2goal = 10000
        # self.d2wp = 8

    # Resets environment for new episode
    def reset(self):
        ##########################3
        self.map = self.world.get_map()
        self.d2goal = 10000
        # Initialize the route planner
        self.dao = GlobalRoutePlannerDAO(self.map, 2.0)  # Create a route for every 2m
        self.grp = GlobalRoutePlanner(self.dao)
        self.grp.setup()
        ###################

        # Car, sensors, etc. We create them every episode then destroy
        self.actor_list = []

        # When Carla breaks (stops working) or spawn point is already occupied, spawning a car throws an exception
        # We allow it to try for 3 seconds then forgive (will cause episode restart and in case of Carla broke inform
        # main thread that Carla simulator needs a restart)
        spawn_start = time.time()
        while True:
            try:
                # Get random spot from a list from predefined spots and try to spawn a car there
                # self.transform = random.choice(self.world.get_map().get_spawn_points())
                # self.vehicle = self.world.spawn_actor(self.mycar, self.transform)

                spawn_points = self.map.get_spawn_points()

                while self.d2goal > 150 or self.d2goal < 50:
                    pos_a = random.choice(spawn_points)
                    pos_b = random.choice(spawn_points)

                    a = pos_a.location
                    b = pos_b.location

                    self.current_plan = self.grp.trace_route(a, b)

                    self.d2goal = self.total_distance(self.current_plan)

                self.transform = pos_a
                self.vehicle = self.world.try_spawn_actor(self.mycar, self.transform)

                # self.transform = random.choice(self.world.get_map().get_spawn_points())
                # self.vehicle = self.world.spawn_actor(self.mycar, self.transform)
                # self.vehicle.set_transform(pos_a)

                args_lateral_dict = {
                    'K_P': 2,  # 1
                    'K_D': 0.2,  # 0.02
                    'K_I': 1,
                    'dt': 1.0 / 20.0}
                target_speed = 50
                self._local_planner = ModifiedLocalPlanner(
                    self.vehicle, opt_dict={'target_speed': target_speed,
                                           'lateral_control_dict': args_lateral_dict})

                assert self.current_plan
                self._local_planner.set_global_plan(self.current_plan)
                self.current_plan = self.current_plan[0:len(self.current_plan) - 5][:]

                # self.d2goal = self.total_distance(self.current_plan)
                # self.d2wp =  self.total_distance(self.current_plan[0:1][:])
                # self.goal = self.current_plan[-1][0]
                # self.start_point = self.current_plan[1][0]

                # Draw path for debugging
                self.draw_path(self.world, self.current_plan)

                break
            except:
                time.sleep(0.01)

            # If that can't be done in 3 seconds - forgive (and allow main process to handle for this problem)
            if time.time() > spawn_start + 3:
                raise Exception('Can\'t spawn a car')

        # Append actor to a list of spawned actors, we need to remove them later
        self.actor_list.append(self.vehicle)

        # Get the blueprint for the camera
        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Set sensor resolution and field of view
        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.rgb_cam.set_attribute('fov', '110')

        # Set camera sensor relative to a car
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))

        # Attach camera sensor to a car, so it will keep relative difference to it
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)

        # Register a callback called every time sensor sends a new data
        self.sensor.listen(self._process_img)

        # Add camera sensor to the list of actors
        self.actor_list.append(self.sensor)

        # Preview ("above the car") camera
        if self.preview_camera_enabled is not False:
            # Get the blueprint for the camera
            self.preview_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
            # Set sensor resolution and field of view
            self.preview_cam.set_attribute('image_size_x', f'{self.preview_camera_enabled[0]:0f}')
            self.preview_cam.set_attribute('image_size_y', f'{self.preview_camera_enabled[1]:0f}')
            self.preview_cam.set_attribute('fov', '110')

            # Set camera sensor relative to a car
            transform = carla.Transform(carla.Location(x=self.preview_camera_enabled[2], y=self.preview_camera_enabled[3], z=self.preview_camera_enabled[4]))

            # Attach camera sensor to a car, so it will keep relative difference to it
            self.preview_sensor = self.world.spawn_actor(self.preview_cam, transform, attach_to=self.vehicle)

            # Register a callback called every time sensor sends a new data
            self.preview_sensor.listen(self._process_preview_img)

            # Add camera sensor to the list of actors
            self.actor_list.append(self.preview_sensor)

        # Here's first workaround. If we do not apply any control it takes almost a second for car to start moving
        # after episode restart. That starts counting once we aplly control for a first time.
        # Workarounf here is to apply both throttle and brakes and disengage brakes once we are ready to start an episode.
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=1.0))

        # And here's another workaround - it takes a bit over 3 seconds for Carla simulator to start accepting any
        # control commands. Above time adds to this one, but by those 2 tricks we start driving right when we start an episode
        # But that adds about 4 seconds of a pause time between episodes.
        time.sleep(4)

        # Collision history is a list callback is going to append to (we brake simulation on a collision)
        self.collision_hist = []

        # Get the blueprint of the sensor
        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')

        # Create the collision sensor and attach ot to a car
        self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(), attach_to=self.vehicle)

        # Register a callback called every time sensor sends a new data
        self.colsensor.listen(self._collision_data)

        # Add the collision sensor to the list of actors
        self.actor_list.append(self.colsensor)

        # Almost ready to start an episode, reset camera update variable
        self.last_cam_update = time.time()

        # Wait for a camera to send first image (important at the beginning of first episode)
        while self.front_camera is None or (self.preview_camera_enabled is not False and self.preview_camera is None):
            time.sleep(0.01)

        # Disengage brakes
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0))

        # Remember a time of episode start (used to measure duration and set a terminal state)
        self.episode_start = time.time()

        ## Add one vehicle on our specific route
        obs_wp = self.current_plan[math.ceil(len(self.current_plan) / 1.3)][0]
        car_actor = self.world.try_spawn_actor(self.obs, pos_b)
        car_actor.set_transform(obs_wp.transform)

        # We set the vehicle on the road, with random moving according to a discrete uniform distribution
        car_actor = self.world.try_spawn_actor(self.obs, pos_b)

        if random.randint(0,10) < 5:
            obs_wp = self.current_plan[math.ceil(len(self.current_plan) / 3)][0]
            car_actor.set_autopilot()
        else:
            obs_wp = self.current_plan[math.ceil(len(self.current_plan) / 2)][0]

        car_actor.set_transform(obs_wp.transform)


        self.actor_list.append(car_actor) # No le pongo sensor porque a este pobre solo lo voy a usar como prop


        # # Get the blueprint of the sensor, I don't care about the poor other car, no col sensor
        # colsens = self.world.get_blueprint_library().find('sensor.other.collision')
        #
        # # Create the collision sensor and attach it to the car
        # colsensor = self.world.spawn_actor(colsens, carla.Transform(), attach_to=car_actor)
        #
        # # Register a callback called every time sensor sends a new data
        # colsensor.listen(self._collision_data)

        # Add the car and collision sensor to the list of car NPCs
        # self.actor_list.append(self.colsensor)

        # Return first observation space - current image from the camera sensor
        return [self.front_camera, 0]

    # Collision data callback handler
    def _collision_data(self, event):

        # What we collided with and what was the impulse
        collision_actor_id = event.other_actor.type_id
        collision_impulse = math.sqrt(event.normal_impulse.x ** 2 + event.normal_impulse.y ** 2 + event.normal_impulse.z ** 2)

        # Filter collisions
        for actor_id, impulse in settings.COLLISION_FILTER:
            if actor_id in collision_actor_id and (impulse == -1 or collision_impulse <= impulse):
                return

        # Add collision
        self.collision_hist.append(event)

    # Camera sensor data callback handler
    def _process_img(self, image):

        # Get image, reshape and drop alpha channel
        image = np.array(image.raw_data)
        image = image.reshape((self.im_height, self.im_width, 4))
        image = image[:, :, :3]

        # Set as a current frame in environment
        self.front_camera = image

        # Measure frametime using a time of last camera update (displayed as Carla FPS)
        if self.playing:
            self.frametimes.append(time.time() - self.last_cam_update)
        else:
            self.frametimes.put_nowait(time.time() - self.last_cam_update)
        self.last_cam_update = time.time()

    # Preview camera sensor data callback handler
    def _process_preview_img(self, image):

        # If camera is disabled - do not process images
        if self.preview_camera_enabled is False:
            return

        # Get image, reshape and drop alpha channel
        image = np.array(image.raw_data)
        try:
            image = image.reshape((int(self.preview_camera_enabled[1]), int(self.preview_camera_enabled[0]), 4))
        except:
            return
        image = image[:, :, :3]

        # Set as a current frame in environment
        self.preview_camera = image

    # Steps environment
    def step(self, action):

        # Monitor if carla stopped sending images for longer than a second. If yes - it broke
        if time.time() > self.last_cam_update + 1:
            raise Exception('Missing updates from Carla')

        # Apply control to the vehicle based on an action
        if self.actions[action] != ACTIONS.no_action:
            self.vehicle.apply_control(carla.VehicleControl(throttle=ACTION_CONTROL[self.actions[action]][0], steer=ACTION_CONTROL[self.actions[action]][2]*self.STEER_AMT, brake=ACTION_CONTROL[self.actions[action]][1]))

        # Calculate speed in km/h from car's velocity (3D vector)
        v = self.vehicle.get_velocity()
        kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

        # done = False
        #
        # # If car collided - end and episode and send back a penalty
        # if len(self.collision_hist) != 0:
        #     done = True
        #     reward = -1
        #
        # # Reward
        # elif settings.WEIGHT_REWARDS_WITH_SPEED == 'discrete':
        #     reward = settings.SPEED_MIN_REWARD if kmh < 50 else settings.SPEED_MAX_REWARD
        #
        # elif settings.WEIGHT_REWARDS_WITH_SPEED == 'linear':
        #     reward = kmh * (settings.SPEED_MAX_REWARD - settings.SPEED_MIN_REWARD) / 100 + settings.SPEED_MIN_REWARD
        #
        # elif settings.WEIGHT_REWARDS_WITH_SPEED == 'quadratic':
        #     reward = (kmh / 100) ** 1.3 * (settings.SPEED_MAX_REWARD - settings.SPEED_MIN_REWARD) + settings.SPEED_MIN_REWARD

        # d2wp, self.d2goal, wp_in_line = self.world._local_planner.run_step(debug=False)
        d2wp = 10
        self.d2goal = 100

        done = False

        eps = 5
        d0 = 8
        kmh0 = 50

        # If car collided - end and episode and send back a penalty
        if len(self.collision_hist) != 0:
            done = True
            reward = -1

        elif self.d2goal > eps:
            reward = 1 - self.d2goal / self.prev_d2goal - d2wp / d0 + kmh / kmh0

        elif self.d2goal <= eps:
            done = True
            reward = 100  # original 1

        # If episode duration limit reached - send back a terminal state
        if not self.playing and self.episode_start + self.seconds_per_episode.value < time.time():
            done = True

        # Weights rewards (not for terminal state)
        if not self.playing and settings.WEIGHT_REWARDS_WITH_EPISODE_PROGRESS and not done:
            reward *= (time.time() - self.episode_start) / self.seconds_per_episode.value

        self.prev_d2goal = self.d2goal

        return [self.front_camera, kmh, self.d2goal, d2wp], reward, done, None

    # Destroys all agents created from last .reset() call
    def destroy_agents(self):

        for actor in self.actor_list:

            # If it has a callback attached, remove it first
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()

            # If it's still alive - destroy it
            if actor.is_alive:
                actor.destroy()

        self.actor_list = []


operating_system = operating_system()


# Returns binary
def get_binary():
    return 'CarlaUE4.exe' if operating_system == 'windows' else 'CarlaUE4.sh'


# Returns exec command
def get_exec_command():
    binary = get_binary()
    exec_command = binary if operating_system == 'windows' else ('./' + binary)

    return binary, exec_command


# tries to close, and if that does not work to kill all carla processes
def kill_processes():

    binary = get_binary()

    # Iterate processes and terminate carla ones
    for process in psutil.process_iter():
        if process.name().lower().startswith(binary.split('.')[0].lower()):
            try:
                process.terminate()
            except:
                pass

    # Check if any are still alive, create a list
    still_alive = []
    for process in psutil.process_iter():
        if process.name().lower().startswith(binary.split('.')[0].lower()):
            still_alive.append(process)

    # Kill process and wait until it's being killed
    if len(still_alive):
        for process in still_alive:
            try:
                process.kill()
            except:
                pass
        psutil.wait_procs(still_alive)


# Starts Carla simulator
def start(playing=False):
    # Kill Carla processes if there are any and start simulator
    if settings.CARLA_HOSTS_TYPE == 'local':
        print('Starting Carla...')
        kill_processes()
        for process_no in range(1 if playing else settings.CARLA_HOSTS_NO):
            # subprocess.Popen(get_exec_command()[1] + f' -carla-rpc-port={settings.CARLA_HOSTS[process_no][1]}', cwd=settings.CARLA_PATH, shell=True)
            subprocess.Popen('SDL_VIDEODRIVER=offscreen ' + get_exec_command()[1] + f' -carla-rpc-port={settings.CARLA_HOSTS[process_no][1]}', cwd=settings.CARLA_PATH, shell=True)
            time.sleep(2)

    # Else just wait for it to be ready
    else:
        print('Waiting for Carla...')

    # Wait for Carla Simulator to be ready
    for process_no in range(1 if playing else settings.CARLA_HOSTS_NO):
        while True:
            try:
                client = carla.Client(*settings.CARLA_HOSTS[process_no][:2])
                # settings.no_rendering_mode = True
                map_name = client.get_world().get_map().name
                if len(settings.CARLA_HOSTS[process_no]) == 2 or not settings.CARLA_HOSTS[process_no][2]:
                    break
                if isinstance(settings.CARLA_HOSTS[process_no][2], int):
                    map_choice = random.choice([map.split('/')[-1] for map in client.get_available_maps()])
                else:
                    map_choice = settings.CARLA_HOSTS[process_no][2]
                if map_name != map_choice:
                    carla.Client(*settings.CARLA_HOSTS[process_no][:2]).load_world(map_choice)
                    while True:
                        try:
                            while carla.Client(*settings.CARLA_HOSTS[process_no][:2]).get_world().get_map().name != map_choice:
                                time.sleep(0.1)
                            break
                        except:
                            pass
                break
            except Exception as e:
                #print(str(e))
                time.sleep(0.1)


# Retarts Carla simulator
def restart(playing=False):
    # Kill Carla processes if there are any and start simulator
    if settings.CARLA_HOSTS_TYPE == 'local':
        for process_no in range(1 if playing else settings.CARLA_HOSTS_NO):
            # subprocess.Popen(get_exec_command()[1] + f' -carla-rpc-port={settings.CARLA_HOSTS[process_no][1]}', cwd=settings.CARLA_PATH, shell=True)
            subprocess.Popen('SDL_VIDEODRIVER=offscreen ' + get_exec_command()[1] + f' -carla-rpc-port={settings.CARLA_HOSTS[process_no][1]}', cwd=settings.CARLA_PATH, shell=True)
            time.sleep(2)

    # Wait for Carla Simulator to be ready
    for process_no in range(1 if playing else settings.CARLA_HOSTS_NO):
        retries = 0
        while True:
            try:
                client = carla.Client(*settings.CARLA_HOSTS[process_no][:2])
                map_name = client.get_world().get_map().name
                if len(settings.CARLA_HOSTS[process_no]) == 2 or not settings.CARLA_HOSTS[process_no][2]:
                    break
                if isinstance(settings.CARLA_HOSTS[process_no][2], int):
                    map_choice = random.choice([map.split('/')[-1] for map in client.get_available_maps()])
                else:
                    map_choice = settings.CARLA_HOSTS[process_no][2]
                if map_name != map_choice:
                    carla.Client(*settings.CARLA_HOSTS[process_no][:2]).load_world(map_choice)
                    while True:
                        try:
                            while carla.Client(*settings.CARLA_HOSTS[process_no][:2]).get_world().get_map().name != map_choice:
                                time.sleep(0.1)
                                retries += 1
                                if retries >= 60:
                                    raise Exception('Couldn\'t change map [1]')
                            break
                        except Exception as e:
                            time.sleep(0.1)
                        retries += 1
                        if retries >= 60:
                            raise Exception('Couldn\'t change map [2]')

                break
            except Exception as e:
                #print(str(e))
                time.sleep(0.1)

            retries += 1
            if retries >= 60:
                break


# Parts of weather control code and npc car spawn code are copied from dynamic_weather.py and spawn_npc.py from examples, then modified
def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        self.altitude = 35.0 * (math.sin(self._t) + 1.0)


class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.puddles = 0.0
        self.wind = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 75.0)
        self.wind = clamp(self._t - delay, 0.0, 80.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False


class Weather(object):
    def __init__(self, weather):
        self.weather = weather
        self.sun = Sun(weather.sun_azimuth_angle, weather.sun_altitude_angle)
        self.storm = Storm(weather.precipitation)

    def set_new_weather(self, weather):
        self.weather = weather

    def tick(self, delta_seconds):
        delta_seconds += random.uniform(-0.1, 0.1)
        self.sun.tick(delta_seconds)
        self.storm.tick(delta_seconds)
        self.weather.cloudyness = self.storm.clouds
        self.weather.precipitation = self.storm.rain
        self.weather.precipitation_deposits = self.storm.puddles
        self.weather.wind_intensity = self.storm.wind
        self.weather.sun_azimuth_angle = self.sun.azimuth
        self.weather.sun_altitude_angle = self.sun.altitude


# Carla settings states
@dataclass
class CARLA_SETTINGS_STATE:
    starting = 0
    working = 1
    restarting = 2
    finished = 3
    error = 4


# Carla settings state messages
CARLA_SETTINGS_STATE_MESSAGE = {
    0: 'STARTING',
    1: 'WORKING',
    2: 'RESTARTING',
    3: 'FINISHED',
    4: 'ERROR',
}

# Carla settings class
class CarlaEnvSettings:

    def __init__(self, process_no, agent_pauses, stop=None, car_npcs=[0, 0], stats=[0., 0., 0., 0., 0., 0.]):

        # Speed factor changes how fast weather should change
        self.speed_factor = 1.0

        # Process number (Carla instance to use)
        self.process_no = process_no

        # Weather and NPC variables
        self.weather = None
        self.spawned_car_npcs = {}

        # Set stats (for Tensorboard)
        self.stats = stats

        # Set externally to restarts settings
        self.restart = False

        # Controls number of NPCs and reset interval
        self.car_npcs = car_npcs

        # State for stats
        self.state = CARLA_SETTINGS_STATE.starting

        # External stop object (used to "know" when to exit
        self.stop = stop

        # We want to track NPC collisions so we can remove and spawn new ones
        # Collisions are really not rare when using built-in autopilot
        self.collisions = Queue()

        # Name of current world
        self.world_name = None

        # Controls world reloads
        self.reload_world_every = None if len(settings.CARLA_HOSTS[process_no]) == 2 or not settings.CARLA_HOSTS[process_no][2] or not isinstance(settings.CARLA_HOSTS[process_no][2], int) else (settings.CARLA_HOSTS[process_no][2] + random.uniform(-settings.CARLA_HOSTS[process_no][2]/10, settings.CARLA_HOSTS[process_no][2]/10))*60
        self.next_world_reload = None if self.reload_world_every is None else time.time() + self.reload_world_every

        # List of communications objects allowing Carla to pause agents (on changes like world change)
        self.agent_pauses = agent_pauses

    # Collect NPC collision data
    def _collision_data(self, collision):
        self.collisions.put(collision)

    # Destroys given car NPC
    def _destroy_car_npc(self, car_npc):

        # First check if NPC is still alive
        if car_npc in self.spawned_car_npcs:

            # Iterate all agents (currently car itself and collision sensor)
            for actor in self.spawned_car_npcs[car_npc]:

                # If actor has any callback attached - stop it
                if hasattr(actor, 'is_listening') and actor.is_listening:
                    actor.stop()

                # And if is still alive - destroy it
                if actor.is_alive:
                    actor.destroy()

            # Remove from car NPCs' list
            del self.spawned_car_npcs[car_npc]

    def clean_carnpcs(self):

        # If there were any NPC cars - remove attached callbacks from it's agents
        for car_npc in self.spawned_car_npcs.keys():
            for actor in self.spawned_car_npcs[car_npc]:
                if hasattr(actor, 'is_listening') and actor.is_listening:
                    actor.stop()

        # Reset NPC car list
        self.spawned_car_npcs = {}

    # Main method, being run in a thread
    def update_settings_in_loop(self):

        # Reset world name
        self.world_name = None

        # Reset weather object
        self.weather = None

        # Run infinitively
        while True:

            # Release agent pause locks, if there are any
            for agent_pause in self.agent_pauses:
                agent_pause.value = 0

            # Carla might break, make sure we can handle for that
            try:

                # If stop flag - exit
                if self.stop is not None and self.stop.value == STOP.stopping:
                    self.state = CARLA_SETTINGS_STATE.finished
                    return

                # If restart flag is being set - wait
                if self.restart:
                    self.state = CARLA_SETTINGS_STATE.restarting
                    time.sleep(0.1)
                    continue

                # Clean car npcs
                self.clean_carnpcs()

                # Connect to Carla, get worls and map
                self.client = carla.Client(*settings.CARLA_HOSTS[self.process_no][:2])
                self.client.set_timeout(2.0)
                self.world = self.client.get_world()
                self.map = self.world.get_map()
                self.world_name = self.map.name

                # Create weather object or update it if exists
                if self.weather is None:
                    self.weather = Weather(self.world.get_weather())
                else:
                    self.weather.set_new_weather(self.world.get_weather())

                # Get car blueprints and filter them
                self.car_blueprints = self.world.get_blueprint_library().filter('vehicle.*')
                self.car_blueprints = [x for x in self.car_blueprints if int(x.get_attribute('number_of_wheels')) == 4]
                self.car_blueprints = [x for x in self.car_blueprints if not x.id.endswith('isetta')]
                self.car_blueprints = [x for x in self.car_blueprints if not x.id.endswith('carlacola')]

                # Get a list of all possible spawn points
                self.spawn_points = self.map.get_spawn_points()

                # Get collision sensor blueprint
                self.collision_sensor = self.world.get_blueprint_library().find('sensor.other.collision')

                # Used to know when to reset next NPC car
                car_despawn_tick = 0

                # Set state to working
                self.state = CARLA_SETTINGS_STATE.working

            # In case of error, report it, wait a second and try again
            except Exception as e:
                self.state = CARLA_SETTINGS_STATE.error
                time.sleep(1)
                continue

            # Steps all settings
            while True:

                # Used to measure sleep time at the loop end
                step_start = time.time()

                # If stop flag - exit
                if self.stop is not None and self.stop.value == STOP.stopping:
                    self.state = CARLA_SETTINGS_STATE.finished
                    return

                # Is restart flag is being set, break inner loop
                if self.restart:
                    break

                # Carla might break, make sure we can handle for that
                try:

                    # World reload
                    if self.next_world_reload and time.time() > self.next_world_reload:

                        # Set restart flag
                        self.state = CARLA_SETTINGS_STATE.restarting
                        # Clean car npcs
                        self.clean_carnpcs()

                        # Set pause lock flag
                        for agent_pause in self.agent_pauses:
                            agent_pause.value = 1

                        # Wait for agents to stop playing and acknowledge
                        for agent_pause in self.agent_pauses:
                            while agent_pause.value != 2:
                                time.sleep(0.1)

                        # Get random map and load it
                        map_choice = random.choice(list({map.split('/')[-1] for map in self.client.get_available_maps()} - {self.client.get_world().get_map().name}))
                        self.client.load_world(map_choice)

                        # Wait for world to be fully loaded
                        retries = 0
                        while self.client.get_world().get_map().name != map_choice:
                            retries += 1
                            if retries >= 600:
                                raise Exception('Timeout when waiting for new map to be fully loaded')
                            time.sleep(0.1)

                        # Inform agents that they can start playing
                        for agent_pause in self.agent_pauses:
                            agent_pause.value = 3

                        # Wait for agents to start playing
                        for agent_pause in self.agent_pauses:
                            retries = 0
                            while agent_pause.value != 0:
                                retries += 1
                                if retries >= 600:
                                    break
                            time.sleep(0.1)

                        self.next_world_reload += self.reload_world_every

                        break

                    # Handle all registered collisions
                    while not self.collisions.empty():

                        # Gets first collision from the queue
                        collision = self.collisions.get()

                        # Gets car NPC's id and destroys it
                        car_npc = collision.actor.id
                        self._destroy_car_npc(car_npc)

                    # Count tick
                    car_despawn_tick += 1

                    # Carla autopilot might cause cars to stop in the middle of intersections blocking whole traffic
                    # On some intersections there might be only one car moving
                    # We want to check for cars stopped at intersections and remove them
                    # Without that most of the cars can be waiting around 2 intersections
                    for car_npc in self.spawned_car_npcs.copy():

                        # First check if car is moving
                        # It;s a simple check, not proper velocity calculation
                        velocity = self.spawned_car_npcs[car_npc][0].get_velocity()
                        simple_speed = velocity.x + velocity.y + velocity.z

                        # If car is moving, continue loop
                        if simple_speed > 0.1 or simple_speed < -0.1:
                            continue

                        # Next get current location of the car, then a waypoint then check if it's intersection
                        location = self.spawned_car_npcs[car_npc][0].get_location()
                        waypoint = self.map.get_waypoint(location)
                        if not waypoint.is_intersection:
                            continue

                        # Car is not moving, it's intersection - destroy a car
                        self._destroy_car_npc(car_npc)

                    # If we reached despawn tick, remove oldest NPC
                    # The reson we want to do that is to rotate cars aroubd the map
                    if car_despawn_tick >= self.car_npcs[1] and len(self.spawned_car_npcs):

                        # Get id of the first car on a list and destroy it
                        car_npc = list(self.spawned_car_npcs.keys())[0]
                        self._destroy_car_npc(car_npc)
                        car_despawn_tick = 0

                    # If there is less number of car NPCs then desired amount - spawn remaining ones
                    # but up to 10 at the time
                    if len(self.spawned_car_npcs) < self.car_npcs[0]:

                        # How many cars to spawn (up to 10)
                        cars_to_spawn = min(10, self.car_npcs[0] - len(self.spawned_car_npcs))

                        # Sometimes we can;t spawn a car
                        # It might be because spawn point is being occupied or because Carla broke
                        # We count errores and break on 5
                        retries = 0

                        # Iterate over number of cars to spawn
                        for _ in range(cars_to_spawn):

                            # Break if too many errors
                            if retries >= 5:
                                break

                            # Get random car blueprint and randomize color and enable autopilot
                            car_blueprint = random.choice(self.car_blueprints)
                            if car_blueprint.has_attribute('color'):
                                color = random.choice(car_blueprint.get_attribute('color').recommended_values)
                                car_blueprint.set_attribute('color', color)
                            car_blueprint.set_attribute('role_name', 'autopilot')

                            # Try to spawn a car
                            for _ in range(5):
                                try:
                                    # Get random spot from a list from predefined spots and try to spawn a car there
                                    spawn_point = random.choice(self.spawn_points)
                                    car_actor = self.world.spawn_actor(car_blueprint, spawn_point)
                                    car_actor.set_autopilot()
                                    break
                                except:
                                    retries += 1
                                    time.sleep(0.1)
                                    continue

                            # Create the collision sensor and attach it to the car
                            colsensor = self.world.spawn_actor(self.collision_sensor, carla.Transform(), attach_to=car_actor)

                            # Register a callback called every time sensor sends a new data
                            colsensor.listen(self._collision_data)

                            # Add the car and collision sensor to the list of car NPCs
                            self.spawned_car_npcs[car_actor.id] = [car_actor, colsensor]

                    # Tick a weather and set it in Carla
                    self.weather.tick(self.speed_factor)
                    self.world.set_weather(self.weather.weather)

                    # Set stats for tensorboard
                    self.stats[0] = len(self.spawned_car_npcs)
                    self.stats[1] = self.weather.sun.azimuth
                    self.stats[2] = self.weather.sun.altitude
                    self.stats[3] = self.weather.storm.clouds
                    self.stats[4] = self.weather.storm.wind
                    self.stats[5] = self.weather.storm.rain

                    # In case of state being some other one report that everything is working
                    self.state = CARLA_SETTINGS_STATE.working

                    # Calculate how long to sleep and sleep
                    sleep_time = self.speed_factor - time.time() + step_start
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                # In case of error, report it (reset flag set externally might break this loop only)
                except Exception as e:
                    #print(str(e))
                    self.state = CARLA_SETTINGS_STATE.error
