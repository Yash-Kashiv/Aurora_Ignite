#!/usr/bin/env python3

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
        self.x_target = np.array([0.45, 0.0, 0.45])   # meters
        self.q_target = np.array([0.9983181956026925, 0.03432997348545016, 0.03850373764272704, 0.026451756777431213])  # quaternion (xyzw)

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
            joint_state_topic="/fr3/joint_states",
            ee_pose_topic="/fr3/ee_pose",
            ee_pose_msg_type=Pose,
            velocity_command_topic="/fr3/joint_velocity_controller/joint_velocity_command",
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
