#!/usr/bin/env python3

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

        self.direction = np.array([0.0, -1.0, 0.0])  # −Y direction
        self.length = 0.30                            # meters
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
            joint_state_topic="/fr3/joint_states",
            ee_pose_topic="/fr3/ee_pose",
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

        # Line equation: x(s) = x0 + s * d
        x_target = self.x_start + s * self.direction
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
