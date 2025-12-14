#!/usr/bin/env python3
"""
Authors: Samriddhi Dubey, MTech, IIT Gandhinagar
         Yash Kashiv, MTech, IIT Gandhinagar

REFERENCE SOLUTION: Inverse Kinematics Challenge

This code implements inverse kinematics for the FR3 robotic arm using Damped Least Squares (DLS) method.
It controls the end-effector to reach a target position and orientation in Cartesian space by:
1. Computing the error between current and target pose
2. Converting the Cartesian error to desired twist (linear + angular velocity)
3. Using DLS inverse kinematics to compute joint velocities
4. Publishing joint velocity commands to move the robot

The robot uses a proportional controller with position and orientation error thresholds
to smoothly reach the target pose and stop when converged.

STUDENT TODO:
1. Fill in the target position (x_target) and target orientation (q_target) values below.
   - x_target: 3D position in meters [x, y, z]
   - q_target: Quaternion orientation [x, y, z, w]

2. Find and fill in the correct ROS topic names:
   - Run 'rostopic list' in a terminal to see all available topics
   - Find the topic for subscribing to joint states
   - Find the topic for subscribing to end-effector pose
   - Find the topic for publishing joint velocity commands
   - Replace the three "TODO" strings in the DLSVelocityCommander with the correct topic names
"""

import rospy
import numpy as np
from geometry_msgs.msg import Pose

from ds_control.robot_state   import RobotState
from ds_control.kdl_ik_solver import DLSIKSolver
from ds_control.dls_velocity  import DLSVelocityCommander


class FrankaDLSController:
    def __init__(self):
        rospy.init_node("fr3_dls_ee_pose_controller")

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
        # DLS IK solver
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
        rospy.loginfo("Initial EE pose received")

        # -------------------------------
        # Target EE pose
        # -------------------------------
        # TODO: Students should fill in the target position (x, y, z) in meters
        self.x_target = np.array([
            # TODO: Fill in x position
            # TODO: Fill in y position
            # TODO: Fill in z position
        ])

        # TODO: Students should fill in the target orientation as quaternion (x, y, z, w)
        self.q_target = np.array([
            # TODO: Fill in quaternion x
            # TODO: Fill in quaternion y
            # TODO: Fill in quaternion z
            # TODO: Fill in quaternion w
        ])

        # -------------------------------
        # Thresholds
        # -------------------------------
        self.pos_thresh = 1e-3
        self.ori_thresh = 1e-2

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
            velocity_command_topic="TODO",
            max_cartesian_vel=0.05,
            max_angular_vel=0.2,
        )

    # ------------------------------------------------
    # Cartesian error → desired twist
    # ------------------------------------------------
    def twist_fn(self):
        from scipy.spatial.transform import Rotation

        x = self.fr3_state.ee_pos
        q = self.fr3_state.ee_ori

        # Position error
        ep = self.x_target - x

        # Orientation error (SO(3) log map)
        R = Rotation.from_quat(q)
        Rt = Rotation.from_quat(self.q_target)
        eo = (Rt * R.inv()).as_rotvec()

        # Simple proportional controller
        v = 1.5 * ep
        w = 1.0 * eo

        # Stop when converged
        if np.linalg.norm(ep) < self.pos_thresh and np.linalg.norm(eo) < self.ori_thresh:
            rospy.loginfo_throttle(1.0, "Target pose reached")
            return np.zeros(6)

        return np.hstack([v, w])

    def start(self):
        rospy.loginfo("Starting FR3 DLS controller")
        self.commander.run()


if __name__ == "__main__":
    ctrl = FrankaDLSController()
    ctrl.start()
    rospy.spin()
