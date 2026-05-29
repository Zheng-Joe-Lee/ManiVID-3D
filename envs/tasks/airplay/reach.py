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

_CONFIG_FILE_NAME = 'airplay/reach.json'

def airplay_reach(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Create a ur5e env, aiming to push a cube to a specified location.
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
    task = Reach()
    environment_kwargs = environment_kwargs or {}
    return RandEnvironment(
        physics, task, config, time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP, **environment_kwargs)

class Physics(RandPhysics):

    def end_to_target(self):
        data = self.named.data
        end_to_obj = data.site_xpos['object_site'] - data.site_xpos['tcp_site']
        return np.linalg.norm(end_to_obj)



class Reach(BaseTask):
    """A dense reward lifting task for UR5.
    """
    def __init__(
        self,
        object_low=(0, -0.05, 0.7145),
        object_high=(0.1, 0.05, 0.7145),
        random=None,
        action_delay=0
    ):
        super().__init__(random, action_delay)
        self.object_low = object_low
        self.object_high = object_high

    def initialize_episode(self, physics):
        # physics.set_freejoint_pos('object_anchor', np.random.uniform(
        #     low=self.object_low, high=self.object_high), np.zeros(4))

        super().initialize_episode(physics)
        physics.set_freejoint_pos('object_anchor', np.random.uniform(
            low=self.object_low, high=self.object_high), np.zeros(4))

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.data.qpos[:].copy()
        obs['velocity'] = physics.data.qvel[:].copy()
        return obs

    def get_reward(self, physics):

        new_action = self._rescale_action(physics)
        action_penalty = np.sum(new_action ** 2) / new_action.shape[0]
        # print("action: ", self.current_action)
        # print("new_action: ", new_action)
        # print("penalty: ", action_penalty)
        distance = physics.end_to_target()

        return rewards.tolerance(distance, bounds=(0, 0.01), margin=0.1) - 0.01 * action_penalty
    
    # def get_termination(self, physics):
    #     if physics.check_lift('object_site', margin=0.04):
    #         return 0.0
