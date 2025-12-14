#!/usr/bin/env python3
"""
Authors: Samriddhi Dubey, MTech, IIT Gandhinagar
         Yash Kashiv, MTech, IIT Gandhinagar

This code makes the FR3 robot's end-effector follow a circular trajectory in the YZ plane.
It demonstrates continuous Cartesian space motion control using:
1. Parametric circle generation in the YZ plane using trigonometric functions
2. Real-time trajectory tracking with DLS inverse kinematics
3. Proportional control to minimize position and orientation errors
4. Continuous motion that repeats indefinitely

The circle parameters:
- Center: Current end-effector position at startup
- Radius: 0.15 meters
- Period: 10 seconds per revolution
- Plane: YZ (perpendicular to X-axis)

The end-effector orientation remains fixed while the position traces a circle.
This is useful for testing Cartesian trajectory following and velocity control.
"""

import rospy
import numpy as np
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation

from ds_control.robot_state   import RobotState
from ds_control.kdl_ik_solver import DLSIKSolver
from ds_control.dls_velocity  import DLSVelocityCommander


class FrankaCircleYZController:
    def __init__(self):
        rospy.init_node("fr3_cartesian_circle_yz")

        # -------------------------------
        # Joint names
        # -------------------------------
        self.fr3_joints = [f"fr3_joint{i}" for i in range(1, 8)]

        # -------------------------------
        # Robot state
        # -------------------------------
        self.fr3_state = RobotState(
            name="fr3",
            joint_names=self.fr3_joints,
            logger=rospy
        )

        # -------------------------------
        # IK Solver
        # -------------------------------
        self.ik = DLSIKSolver(
            urdf_param="/fr3/robot_description",
            base_link="fr3_link0",
            tip_link="fr3_link8",
            joint_names=self.fr3_joints,
            damping=0.01
        )

        rospy.loginfo("Waiting for initial EE pose...")
        ee_msg = rospy.wait_for_message("/fr3/ee_pose", Pose)
        self.fr3_state.update_from_pose(ee_msg)

        # -------------------------------
        # Circle definition
        # -------------------------------
        self.center = self.fr3_state.ee_pos.copy()
        self.q_fixed = self.fr3_state.ee_ori.copy()

        self.radius = 0.15           # meters
        self.period = 10.0           # seconds per revolution
        self.omega = 2.0 * np.pi / self.period

        self.start_time = rospy.get_time()

        # -------------------------------
        # Thresholds
        # -------------------------------
        self.pos_thresh = 1e-3
        self.ori_thresh = 1e-2

        rospy.loginfo("Drawing circle in YZ plane")

        # -------------------------------
        # Velocity commander
        # -------------------------------
        self.commander = DLSVelocityCommander(
            robot_state=self.fr3_state,
            ik_solver=self.ik,
            custom_ds=self.twist_fn,
            joint_state_topic="/fr3/joint_states",
            ee_pose_topic="/fr3/ee_pose",
            ee_pose_msg_type=Pose,
            velocity_command_topic="/fr3/joint_velocity_controller/joint_velocity_command",
            max_cartesian_vel=0.05,
            max_angular_vel=0.2,
        )

    # ------------------------------------------------
    # Cartesian circular trajectory
    # ------------------------------------------------
    def twist_fn(self):
        t = rospy.get_time() - self.start_time

        # Parametric circle in YZ plane
        x_target = self.center[0]
        y_target = self.center[1] + self.radius * np.cos(self.omega * t)
        z_target = self.center[2] + self.radius * np.sin(self.omega * t)

        x_target = np.array([x_target, y_target, z_target])
        q_target = self.q_fixed

        x = self.fr3_state.ee_pos
        q = self.fr3_state.ee_ori

        # Errors
        ep = x_target - x
        eo = (Rotation.from_quat(q_target)
              * Rotation.from_quat(q).inv()).as_rotvec()

        # Cartesian proportional control
        v = 2.0 * ep
        w = 1.0 * eo

        return np.hstack([v, w])

    def start(self):
        rospy.loginfo("Starting circular motion")
        self.commander.run()


if __name__ == "__main__":
    ctrl = FrankaCircleYZController()
    ctrl.start()
    rospy.spin()
