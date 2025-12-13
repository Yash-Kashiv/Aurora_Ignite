#!/usr/bin/env python3

import rospy
import actionlib

from franka_gripper.msg import (
    GraspAction, GraspGoal,
    HomingAction, HomingGoal
)

from control_msgs.msg import (
    GripperCommandAction,
    GripperCommandGoal
)


class FrankaGripperClient:
    def __init__(self):
        rospy.init_node("fr3_grasp_client")

        ns = "/fr3/franka_gripper"

        rospy.loginfo("Waiting for Franka gripper action servers...")

        # Correct action types
        self.homing_client = actionlib.SimpleActionClient(
            ns + "/homing", HomingAction)
        self.homing_client.wait_for_server()

        self.gripper_cmd_client = actionlib.SimpleActionClient(
            ns + "/gripper_action", GripperCommandAction)
        self.gripper_cmd_client.wait_for_server()

        self.grasp_client = actionlib.SimpleActionClient(
            ns + "/grasp", GraspAction)
        self.grasp_client.wait_for_server()

        rospy.loginfo("Gripper action servers ready")

    # -------------------------
    # Homing
    # -------------------------
    def home(self):
        rospy.loginfo("Homing gripper...")
        goal = HomingGoal()
        self.homing_client.send_goal(goal)
        self.homing_client.wait_for_result()

    # -------------------------
    # Open gripper (CORRECT)
    # -------------------------
    def open(self, width=0.08, speed=0.05):
        rospy.loginfo("Opening gripper...")

        goal = GripperCommandGoal()

        # IMPORTANT:
        # position = half-width (per finger)
        goal.command.position = width / 2.0
        goal.command.max_effort = 0.0   # 0 → pure move, no grasp

        self.gripper_cmd_client.send_goal(goal)
        self.gripper_cmd_client.wait_for_result()

    # -------------------------
    # Force-controlled grasp
    # -------------------------
    def grasp(self, width=0.03, speed=0.05, force=10.0):
        rospy.loginfo("Grasping object...")

        goal = GraspGoal()
        goal.width = width
        goal.speed = speed
        goal.force = force
        goal.epsilon.inner = 0.05
        goal.epsilon.outer = 0.05

        self.grasp_client.send_goal(goal)
        self.grasp_client.wait_for_result()

        result = self.grasp_client.get_result()

        if result and result.success:
            rospy.loginfo("Grasp successful")
            return True
        else:
            rospy.logwarn("Grasp failed")
            return False


if __name__ == "__main__":
    gripper = FrankaGripperClient()

    rospy.sleep(1.0)

    gripper.home()
    gripper.open()
    rospy.sleep(1.0)
    gripper.grasp()
