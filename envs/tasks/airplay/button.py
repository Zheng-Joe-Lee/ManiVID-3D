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

_CONFIG_FILE_NAME = 'airplay/button.json'

def airplay_button(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
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
    task = Pressbutton()
    environment_kwargs = environment_kwargs or {}
    return RandEnvironment(
        physics, task, config, time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP, **environment_kwargs)

class Physics(RandPhysics):
    def end_to_hover_xy(self):
        data = self.named.data
        end_to_hover = data.site_xpos['hover_site'] - data.site_xpos['tcp_site']
        return np.linalg.norm(end_to_hover[:2])
    
    def btn_to_pushdown(self):
        data = self.named.data
        btn_to_pushdown = data.site_xpos['pushdown_site'] - data.site_xpos['btn_site']
        return np.linalg.norm(btn_to_pushdown)

    def end_to_press_z(self):
        data = self.named.data
        end_to_press = (data.site_xpos['tcp_site'] + np.array([0, 0, 0.02])) - data.site_xpos['btn_site']
        return np.linalg.norm(end_to_press[-1])

class Pressbutton(BaseTask):
    """A dense reward lifting task for UR5.
    """

    def __init__(
        self,
        target_low=(0, -0.05, 0.758),
        target_high=(0.1, 0.05, 0.758),
        random=None,
        action_delay=0
    ):
        super().__init__(random, action_delay)
        self.object_low = target_low
        self.object_high = target_high

    def initialize_episode(self, physics):
        # sample position
        physics.set_body_pos('buttonbox', np.random.uniform(
            low=self.object_low, high=self.object_high))
        super().initialize_episode(physics)


    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.data.qpos[:].copy()
        obs['velocity'] = physics.data.qvel[:].copy()
        return obs

    def get_reward(self, physics):
        end_to_hover_xy = physics.end_to_hover_xy()
        end_to_press_z = physics.end_to_press_z()
        btn_to_pushdown = physics.btn_to_pushdown()

        hover_reward = rewards.tolerance(end_to_hover_xy, bounds=(0, 0.02), margin=0.1)
        if hover_reward > 0.85:
            press_reward = 2 * rewards.tolerance(end_to_press_z, bounds=(0, 0.005), margin=0.14)
        else:
            press_reward = 0

        if press_reward > 1.5:
            button_reward = 3 * rewards.tolerance(btn_to_pushdown, bounds=(0, 0.005), margin=0.02)
        else:
            button_reward = 0
        
        reward = (hover_reward + press_reward + button_reward) / 6

        return reward

    # def _press_reward(self, object_z, target_z=0, min_z=0):
    #     if object_z >= target_z:
    #         return 1.0
    #     elif object_z <= min_z:
    #         return 0.0
    #     else:
    #         return (object_z - min_z) / (target_z - min_z)
    
    # def get_termination(self, physics):
    #     if physics.check_lift('object_site', margin=0.04):
    #         return 0.0
