"""Arm pieces of the concurrent expert: the final arm pose for a target, and the arm path while the base drives.

The professor's order (3 Nov): 1. the base turns in place to face the target (arm still), 2. the base drives while the
arm moves to its final pose, 3. a short Cartesian correction once the base has parked (it never parks exactly where
planned). The shared controllers in ../robotnik_arm_task are imported, not copied: the base controller, the arm's
Cartesian IK and the stop rules stay the same as the sequential expert's.
"""
import os
import sys

import numpy as np
import pinocchio as pin

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robotnik_arm_task"))
from robotnik_arm_control import arm_controller  # noqa: E402


def solve_final_arm_pose(dyn_model, base_pos, base_bearing, q_motors, target, ee_frame_id, shoulder_frame_id,
                         arm_lower, arm_upper, params, dt=0.01, max_iters=3000, tol=1e-3, stall_iters=200):
    """Arm joint angles that put the EE on the target with the base parked at base_pos (x, y, z) facing base_bearing.

    Kinematics only, no simulation: the arm controller's own IK (damped, joint-limit aware, reach-limited) is iterated
    with the base held at its parking spot and the joints following their reference exactly, starting from the
    current joint angles (so the solution is the one the expert's IK would reach from here, no wrapped joints).
    q_motors: all motor angles (4 wheels, 7 arm, 6 fingers); only the arm part changes.

    Returns (q_final (7), EE error left at it (m), iterations). The error is not 0 for targets beyond the reach
    limit: the IK settles at the closest pose on the reach sphere, like the sequential expert does.
    """
    base_ori = np.array([0.0, 0.0, np.sin(base_bearing / 2), np.cos(base_bearing / 2)])   # x, y, z, w like the sim
    q_motors = np.array(q_motors, dtype=float)
    q_des = q_motors[4:11].copy()
    best_q, best_err, best_iter = q_des.copy(), np.inf, 0
    for it in range(max_iters):
        q_pin = dyn_model.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q_motors)), "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        err = np.linalg.norm(target - dyn_model.pin_data.oMf[ee_frame_id].translation)
        if err < best_err - 1e-5:
            best_q, best_err, best_iter = q_motors[4:11].copy(), err, it
        if err < tol or it - best_iter > stall_iters:
            break
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        _, q_des, _ = arm_controller(
            dyn_model, q_pin, q_motors[4:11], np.zeros(7), q_des, q_des, target, ee_pos,
            ee_frame_id, shoulder_frame_id, arm_lower, arm_upper, dt, params, reach_active=True,
        )
        q_motors[4:11] = q_des
    return best_q, best_err, it


def drive_progress(phase, dist_to_goal, dist_at_drive_start):
    """How far the base is through its trip, 0..1: 0 while turning to face the goal, 1 from the end of the drive."""
    if phase == "align":
        return 0.0
    if phase == "drive" and dist_at_drive_start > 0:
        return float(np.clip(1.0 - dist_to_goal / dist_at_drive_start, 0.0, 1.0))
    return 1.0


def arm_path_reference(q_start, q_final, progress, start_at=0.0):
    """Joint reference along the arm path: minimum-jerk from q_start to q_final over drive progress start_at..1.

    Minimum-jerk in progress, not time: the arm arrives with the base whatever the trip length, and starts and ends
    with zero velocity and acceleration (no kick on the base when the arm starts or stops).
    """
    u = float(np.clip((progress - start_at) / max(1.0 - start_at, 1e-6), 0.0, 1.0))
    blend = 10 * u**3 - 15 * u**4 + 6 * u**5
    return q_start + (q_final - q_start) * blend


def arm_path_controller(dyn_model, q_pin, q_arm, qd_arm, q_des_arm, q_ref, dt, params):
    """Joint PD + gravity compensation tracking q_ref, moved towards at most params.max_joint_vel.

    The rate limit only acts when the base covers its trip faster than the arm can follow (short trips).
    Forward kinematics need not be up to date. Returns (torque_arm, q_des_arm, qd_des_arm), like arm_controller.
    """
    step = q_ref - q_des_arm
    max_step = params.max_joint_vel * dt
    if np.max(np.abs(step)) > max_step:
        step *= max_step / np.max(np.abs(step))
    qd_des_arm = step / dt
    q_des_arm = q_des_arm + step
    torque_arm = params.kp * (q_des_arm - q_arm) + params.kd * (qd_des_arm - qd_arm)
    tau_g = pin.computeGeneralizedGravity(dyn_model.pin_model, dyn_model.pin_data, q_pin)
    torque_arm = np.clip(torque_arm + tau_g[6:13], -params.max_torque, params.max_torque)
    return torque_arm, q_des_arm, qd_des_arm
