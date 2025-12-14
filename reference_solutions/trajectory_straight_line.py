#!/usr/bin/env python3
"""
Authors: Samriddhi Dubey, MTech, IIT Gandhinagar
         Yash Kashiv, MTech, IIT Gandhinagar

REFERENCE SOLUTION: Straight Line Trajectory Challenge

This code makes the FR3 robot's end-effector follow a straight-line trajectory in Cartesian space.
It demonstrates linear motion control using:
1. Parametric line generation using direction vectors
2. Real-time trajectory tracking with DLS inverse kinematics
3. Progress-based motion along a specified direction and distance
4. Automatic stopping when the line endpoint is reached

The line parameters:
- Start point: Current end-effector position at startup
- Direction: A normalized 3D vector (e.g., [0, -1, 0] for -Y direction)
- Length: Distance to travel in meters
- Speed: Constant velocity along the line (0.05 m/s)

The end-effector orientation remains fixed while the position moves along a straight line.
This is useful for testing Cartesian linear motion and velocity control.

STUDENT TODO:
1. Find and fill in the correct ROS topic names:
   - Run 'rostopic list' in a terminal to see all available topics
   - Find the topic for subscribing to joint states
   - Find the topic for subscribing to end-effector pose
   - Replace the two "TODO" strings in the DLSVelocityCommander with the correct topic names

2. Fill in the straight line parameters:

3. Complete the line equation:
   - Fill in the TODO to complete this equation
"""

import rospy
import numpy as np
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation

from ds_control.robot_state   import RobotState
from ds_control.kdl_ik_solver import DLSIKSolver
from ds_control.dls_velocity  import DLSVelocityCommander


class FrankaStraightLineController:
    def __init__(self):
        rospy.init_node("fr3_cartesian_line_controller")

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
        # Line definition
        # -------------------------------
        self.x_start = self.fr3_state.ee_pos.copy()
        self.q_fixed = self.fr3_state.ee_ori.copy()

        self.direction = # TODO  # −Y direction
        self.length = # TODO                            # meters
        self.speed = 0.05                             # m/s

        self.start_time = rospy.get_time()

        # -------------------------------
        # Thresholds
        # -------------------------------
        self.pos_thresh = 1e-3
        self.ori_thresh = 1e-2

        rospy.loginfo("Starting straight-line motion")

        # -------------------------------
        # Velocity commander
        # -------------------------------
        self.commander = DLSVelocityCommander(
            robot_state=self.fr3_state,
            ik_solver=self.ik,
            custom_ds=self.twist_fn,
            joint_state_topic="TODO",
            ee_pose_topic="TODO",
            ee_pose_msg_type=Pose,
            velocity_command_topic="/fr3/joint_velocity_controller/joint_velocity_command",
            max_cartesian_vel=0.05,
            max_angular_vel=0.2,
        )

    # ------------------------------------------------
    # Cartesian straight-line controller
    # ------------------------------------------------
    def twist_fn(self):
        t = rospy.get_time() - self.start_time

        # Scalar progress along the line
        s = min(self.speed * t, self.length)

        # Line equation: 
        x_target = self.x_start # TODO
        q_target = self.q_fixed

        x = self.fr3_state.ee_pos
        q = self.fr3_state.ee_ori

        # Errors
        ep = x_target - x
        eo = (Rotation.from_quat(q_target)
              * Rotation.from_quat(q).inv()).as_rotvec()

        # Stop when end of line is reached
        if s >= self.length and np.linalg.norm(ep) < self.pos_thresh:
            rospy.loginfo_throttle(1.0, "Straight line completed")
            return np.zeros(6)

        # Cartesian velocity command
        v = 2.0 * ep
        w = 1.0 * eo

        return np.hstack([v, w])

    def start(self):
        self.commander.run()


if __name__ == "__main__":
    ctrl = FrankaStraightLineController()
    ctrl.start()
    rospy.spin()
