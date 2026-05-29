import os
import enum
import numpy as np
from dm_control import mjcf
import xml.etree.ElementTree as et

from ..utils import get_mjcf_model
from .robot import Robot

class AirplayWithGripper(Robot):

    _INIT_POSE = {
        Robot.ControlMode.ACTUATOR: {
            'qpos': [0, -0.315, 0.994, -1.57, 0.679, 1.57, 0, 0],
            'ctrl': [0, -0.315, 0.994, -1.57, 0.679, 1.57, 0]
        },
        Robot.ControlMode.MOCAP: {
            'qpos': [0, -0.315, 0.994, -1.57, 0.679, 1.57, 0, 0],
            'ctrl': [0]
        }
    }

    _ARM_JOINT_NAMES = [
        "joint1", "joint2", "joint3", 
        "joint4", "joint5", "joint6"
    ]
    _GRIPPER_JOINT_NAMES = [
        "endleft", "endright"
    ]

    _GRIPPER_JOINT_NUM = 2
    _ARM_JOINT_NUM = 6
