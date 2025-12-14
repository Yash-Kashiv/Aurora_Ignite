#!/usr/bin/env python3
"""
Authors: Samriddhi Dubey, MTech, IIT Gandhinagar
         Yash Kashiv, MTech, IIT Gandhinagar

REFERENCE SOLUTION: Pick and Place Challenge

This code implements a complete pick-and-place operation for the FR3 robotic arm.
It combines inverse kinematics control with gripper manipulation to:
1. Move the end-effector to a home position
2. Navigate to a pick location and grasp an object using the Franka gripper
3. Transport the grasped object to a place location
4. Release the object at the target location
5. Return to the home position

The system uses:
- DLS (Damped Least Squares) inverse kinematics for smooth Cartesian space motion
- Thread-based control for real-time velocity commands
- Action clients for gripper control (homing, opening, grasping)
- Blocking motion primitives to ensure sequential execution of pick-and-place steps

The controller continuously monitors pose errors and stops when position and orientation
thresholds are met, ensuring precise object manipulation.

STUDENT TODO:
1. Find and fill in the correct ROS topic names:
   - Run 'rostopic list' in a terminal to see all available topics
   - Find the topic for subscribing to joint states
   - Find the topic for subscribing to end-effector pose
   - Find the topic for publishing joint velocity commands
   - Replace the three "TODO" strings in the DLSVelocityCommander with the correct topic names

2. Fill in the end-effector poses for the pick-and-place sequence below.
   Each pose consists of:
   - Position: [x, y, z] in meters
   - Orientation: [qx, qy, qz, qw] as a quaternion

"""

import rospy
import numpy as np
import actionlib
import threading
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation

from ds_control.robot_state   import RobotState
from ds_control.kdl_ik_solver import DLSIKSolver
from ds_control.dls_velocity  import DLSVelocityCommander

from franka_gripper.msg import (
    GraspAction, GraspGoal,
    HomingAction, HomingGoal
)
from control_msgs.msg import (
    GripperCommandAction,
    GripperCommandGoal
)


# ============================================================
# Gripper Client
# ============================================================

class FrankaGripperClient:
    def __init__(self):
        ns = "/fr3/franka_gripper"

        rospy.loginfo("Waiting for gripper action servers...")

        self.homing_client = actionlib.SimpleActionClient(
            ns + "/homing", HomingAction)
        self.homing_client.wait_for_server()

        self.gripper_cmd_client = actionlib.SimpleActionClient(
            ns + "/gripper_action", GripperCommandAction)
        self.gripper_cmd_client.wait_for_server()

        self.grasp_client = actionlib.SimpleActionClient(
            ns + "/grasp", GraspAction)
        self.grasp_client.wait_for_server()

        rospy.loginfo("Gripper servers ready")

    def home(self):
        goal = HomingGoal()
        self.homing_client.send_goal(goal)
        self.homing_client.wait_for_result()

    def open(self, width=0.08):
        goal = GripperCommandGoal()
        goal.command.position = width / 2.0  # per finger
        goal.command.max_effort = 0.0
        self.gripper_cmd_client.send_goal(goal)
        self.gripper_cmd_client.wait_for_result()

    def grasp(self, width=0.03, force=10.0):
        goal = GraspGoal()
        goal.width = width
        goal.speed = 0.05
        goal.force = force
        goal.epsilon.inner = 0.05
        goal.epsilon.outer = 0.05

        self.grasp_client.send_goal(goal)
        self.grasp_client.wait_for_result()
        result = self.grasp_client.get_result()
        return result and result.success


# ============================================================
# DLS IK Controller
# ============================================================

class FrankaPickPlaceController:
    def __init__(self):
        rospy.init_node("fr3_pick_and_place")

        self.fr3_joints = [f"fr3_joint{i}" for i in range(1, 8)]

        self.state = RobotState(
            name="fr3",
            joint_names=self.fr3_joints,
            logger=rospy
        )

        self.ik = DLSIKSolver(
            urdf_param="/fr3/robot_description",
            base_link="fr3_link0",
            tip_link="fr3_link8",
            joint_names=self.fr3_joints,
            damping=0.01
        )

        rospy.loginfo("Waiting for initial EE pose...")
        msg = rospy.wait_for_message("/fr3/ee_pose", Pose)
        self.state.update_from_pose(msg)

        self.pos_thresh = 1e-3
        self.ori_thresh = 1e-2

        self.target_lock = threading.Lock()
        self.x_target = self.state.ee_pos.copy()
        self.q_target = self.state.ee_ori.copy()

        self.commander = DLSVelocityCommander(
            robot_state=self.state,
            ik_solver=self.ik,
            custom_ds=self.twist_fn,
            joint_state_topic="TODO",
            ee_pose_topic="TODO",
            ee_pose_msg_type=Pose,
            velocity_command_topic="TODO",
            max_cartesian_vel=0.05,
            max_angular_vel=0.2,
        )

        self.ctrl_thread = threading.Thread(target=self.commander.run)
        self.ctrl_thread.daemon = True
        self.ctrl_thread.start()

        self.gripper = FrankaGripperClient()

    # --------------------------------------------------------
    # Twist function (pure DLS IK)
    # --------------------------------------------------------
    def twist_fn(self):
        with self.target_lock:
            xt = self.x_target
            qt = self.q_target

        x = self.state.ee_pos
        q = self.state.ee_ori

        ep = xt - x
        eo = (Rotation.from_quat(qt) * Rotation.from_quat(q).inv()).as_rotvec()

        if np.linalg.norm(ep) < self.pos_thresh and np.linalg.norm(eo) < self.ori_thresh:
            return np.zeros(6)

        v = 1.5 * ep
        w = 1.0 * eo
        return np.hstack([v, w])

    # --------------------------------------------------------
    # Blocking motion primitive
    # --------------------------------------------------------
    def move_to_pose(self, pos, quat):
        with self.target_lock:
            self.x_target = np.array(pos)
            self.q_target = np.array(quat)

        rospy.loginfo("Moving to target pose...")
        rate = rospy.Rate(50)

        while not rospy.is_shutdown():
            ep = np.linalg.norm(self.state.ee_pos - self.x_target)
            eo = np.linalg.norm(
                (Rotation.from_quat(self.q_target)
                 * Rotation.from_quat(self.state.ee_ori).inv()).as_rotvec()
            )
            if ep < self.pos_thresh and eo < self.ori_thresh:
                rospy.loginfo("Target reached")
                break
            rate.sleep()


# ============================================================
# MAIN SEQUENCE
# ============================================================

if __name__ == "__main__":
    ctrl = FrankaPickPlaceController()
    rospy.sleep(1.0)

    # -------------------------
    # HOME POSE
    # -------------------------
    # TODO: Students should fill in the home pose
    # Position: [x, y, z] in meters
    # Orientation: [qx, qy, qz, qw] as quaternion
    home_pose = (
        [
            # TODO: Fill in home position x
            # TODO: Fill in home position y
            # TODO: Fill in home position z
        ],
        [
            # TODO: Fill in home orientation qx
            # TODO: Fill in home orientation qy
            # TODO: Fill in home orientation qz
            # TODO: Fill in home orientation qw
        ]
    )

    # -------------------------
    # PICK POSE
    # -------------------------
    # TODO: Students should fill in the pick pose (where to grasp the object)
    # Position: [x, y, z] in meters
    # Orientation: [qx, qy, qz, qw] as quaternion
    pick_pose = (
        [
            # TODO: Fill in pick position x
            # TODO: Fill in pick position y
            # TODO: Fill in pick position z
        ],
        [
            # TODO: Fill in pick orientation qx
            # TODO: Fill in pick orientation qy
            # TODO: Fill in pick orientation qz
            # TODO: Fill in pick orientation qw
        ]
    )

    # -------------------------
    # PLACE POSE
    # -------------------------
    # TODO: Students should fill in the place pose (where to release the object)
    # Position: [x, y, z] in meters
    # Orientation: [qx, qy, qz, qw] as quaternion
    place_pose = (
        [
            # TODO: Fill in place position x
            # TODO: Fill in place position y
            # TODO: Fill in place position z
        ],
        [
            # TODO: Fill in place orientation qx
            # TODO: Fill in place orientation qy
            # TODO: Fill in place orientation qz
            # TODO: Fill in place orientation qw
        ]
    )

    # ========================================================
    # EXECUTION
    # ========================================================

    rospy.loginfo("Homing gripper")
    ctrl.gripper.home()
    ctrl.gripper.open()

    ctrl.move_to_pose(*pick_pose)

    rospy.loginfo("Grasping object")
    ctrl.gripper.grasp()
    rospy.sleep(3.0)

    ctrl.move_to_pose(*place_pose)

    rospy.loginfo("Releasing object")
    ctrl.gripper.open()
    rospy.sleep(1.0)

    ctrl.move_to_pose(*home_pose)

    rospy.loginfo("Pick and place completed")
    rospy.spin()

