#!/usr/bin/env python3
"""
Authors: Samriddhi Dubey, MTech, IIT Gandhinagar
         Yash Kashiv, MTech, IIT Gandhinagar

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
            joint_state_topic="/fr3/joint_states",
            ee_pose_topic="/fr3/ee_pose",
            ee_pose_msg_type=Pose,
            velocity_command_topic="/fr3/joint_velocity_controller/joint_velocity_command",
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
    # HOME
    # -------------------------
    home_pose = (
        [0.30759087204933167, 0.00011447259748820215, 0.48617154359817505],
        [0.9999913738392597, 0.004149723519153129,
         6.674362182470983e-06, 0.00017887771648540977]
    )

    # -------------------------
    # PICK
    # -------------------------
    pick_pose = (
        [0.563630998134613, -0.20336730778217316, 0.1382521539926529],
        [0.9995627529155964, 0.028155959387577277,
         -0.008999100017086075, -0.0007490885408360415]
    )

    # -------------------------
    # PLACE
    # -------------------------
    place_pose = (
        [0.5072759389877319, 0.3937574028968811, 0.188519686460495],
        [0.9638487886059453, 0.2658570894324009,
         0.002587464347219151, 0.017573438184830578]
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
