import os
import numpy as np
import collections
import json
from dm_control.utils import rewards
from dm_control.rl import control

from ... import _SUITE_DIR, _AIRPLAY_XML_DIR
from ...robots import AirplayWithGripper
from ..base import BaseTask
from ...randomize.wrapper import RandPhysics, RandEnvironment


_CONTROL_TIMESTEP = .02  # (Seconds)
_DEFAULT_TIME_LIMIT = 10  # Default duration of an episode, in seconds.

_CONFIG_FILE_NAME = 'airplay/laptop.json'

def airplay_laptop(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Create a airplay env, aiming to push a cube to a specified location.
    """
    config_path = os.path.join(_SUITE_DIR, 'configs', _CONFIG_FILE_NAME)
    with open(config_path, mode='r') as f:
        config = json.load(f)
    robot = AirplayWithGripper.from_file_path(
        xml_path=config['xml'],
        asset_paths=config['assets'],
        actuator_path=os.path.join(_AIRPLAY_XML_DIR, 'airbot_play_control.xml'),
        config=config
    )
    physics = Physics.from_rand_mjcf(robot)
    task = Laptopclose()
    environment_kwargs = environment_kwargs or {}
    return RandEnvironment(
        physics, task, config, time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP, **environment_kwargs)

class Physics(RandPhysics):
    def end_to_hover(self):
        data = self.named.data
        end_to_obj = data.site_xpos['laptop_hover_site'] - data.site_xpos['tcp_site']
        return np.linalg.norm(end_to_obj)
    
    def end_to_object(self):
        data = self.named.data
        end_to_obj = data.site_xpos['laptop_site'] - data.site_xpos['tcp_site']
        return np.linalg.norm(end_to_obj)
    
    # def hand_to_object(self):
    #     data = self.named.data
    #     hand_to_obj = data.site_xpos['laptop_hand_site'] - data.site_xpos['tcp_site']
    #     return np.linalg.norm(hand_to_obj)

    def laptop_angle(self):
        """1.57 is the angle when the laptop is closed.
        """
        data = self.named.data
        angle = data.qpos['laptop_joint']
        return angle.item()

class Laptopclose(BaseTask):
    """A dense reward lifting task for UR5.

    """
    def __init__(
        self,
        object_low=(0.05, 0.0, 0.70),
        object_high=(0.15, 0.0, 0.70),
        angle_low=-0.15,
        angle_high=-0.05,
        random=None,
        action_delay=0
    ):
        super().__init__(random, action_delay)
        self.object_low = np.array(object_low)
        self.object_high = np.array(object_high)
        self.angle_low = angle_low
        self.angle_high = angle_high

    def initialize_episode(self, physics):
        # self.delta_table_height = physics.named.data.xpos['table'][2] - 0.0
        
        object_low = self.object_low.copy()
        object_high = self.object_high.copy()
        # object_low[2] += self.delta_table_height
        # object_high[2] += self.delta_table_height
        self.init_angle = np.random.uniform(low=self.angle_low, high=self.angle_high)

        physics.set_body_pos('laptop', np.random.uniform(
            low=object_low, high=object_high))
        physics.set_joint_pos('laptop_joint', self.init_angle)

        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.data.qpos[:].copy()
        obs['velocity'] = physics.data.qvel[:].copy()
        return obs

    def get_reward(self, physics):
        end_to_hover = physics.end_to_hover()
        end_to_obj = physics.end_to_object()
        # hand_to_obj = physics.hand_to_object()
        angle = physics.laptop_angle()

        reward = 0.7 * np.exp(-10 * np.clip(end_to_hover-0.03, 0, None))
        if reward > 0.63:
            reward += 2 * np.exp(-10 * np.clip(end_to_obj-0.03, 0, None))
        # if angle < 0.349:  # 20 degree
        #     reward += 2 * np.exp(-10 * np.clip(hand_to_obj-0.03, 0, None))
        # else:
        #     reward += 2
        
        reward += 3 * (physics.laptop_angle() - self.init_angle)

        if abs(1.57 - physics.laptop_angle()) < 0.05:
            reward += 10

        return reward
    # def get_termination(self, physics):
    #     if physics.check_lift('object_site', margin=0.04):
    #         return 0.0
