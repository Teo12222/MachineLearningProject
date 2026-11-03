"""Concurrent expert: turn to face the target, then drive while the arm moves to its final pose (PLAN.md step 28).

Same simulation, target, base controller, Cartesian arm IK and stop rules as ../robotnik_arm_task/robotnik_arm_task.py
(the sequential expert, where the arm waits for the base to park). What changes is the arm while the base travels:
- align (turn in place): the arm holds its start pose;
- drive: the arm follows a minimum-jerk joint path to q_final, timed by drive progress, so it arrives with the base.
  q_final is solved once at t = 0 (kinematics only) for the base parked at its planned spot;
- orient / parked: the arm finishes the path, then the usual Cartesian IK corrects for where the base really parked.
Saved episodes have the same fields plus q_final, its IK error and the path progress per frame.
"""
import numpy as np
import time
import os

# Headless mode and simulated time cap (s) for CI / direct runs
headless = os.environ.get("ROBOENV_HEADLESS", "").strip().lower() in ("1", "true", "yes")
# max_steps = int(os.environ.get("ROBOENV_MAX_STEPS", "40000"))   # replaced by the time cap, see PLAN.md step 0
max_time = float(os.environ.get("ROBOENV_MAX_TIME", "40"))
out_path = os.environ.get("ROBOENV_OUT", "episode.npz")

import matplotlib
if headless:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as p
import pinocchio as pin
from simulation_and_control import pb, MotorCommands, PinWrapper
# Puts ../robotnik_arm_task on the path too: the shared controllers live with the sequential expert
from concurrent_control import arm_path_controller, arm_path_reference, drive_progress, solve_final_arm_pose
from robotnik_arm_control import (
    ArmControllerParams, BaseControllerParams, BaseNavState, StopCriteria,
    arm_controller, base_controller, check_stop, configure_gripper_fingers, parse_start_pose, quaternion2bearing,
    set_start_pose,
)


# ==============================================================================
# TARGET SPECIFICATION
# Fixed 3D Cartesian coordinates for the Kinova end-effector in the world frame:
# [X (forward/meters), Y (lateral/meters), Z (height/meters)]
_target_str = os.environ.get("ROBOENV_TARGET", "0.0,1.1,0.5")
TARGET_EE_POS = np.array([float(v) for v in _target_str.split(",")])
# Start pose of the base (x m, y m, yaw rad), e.g. ROBOENV_START=1.0,-0.5,1.57. Default: the origin facing +x,
# which all training data uses (PLAN.md step 0, fix 4).
START_POSE = parse_start_pose(os.environ.get("ROBOENV_START", "0,0,0"))
# DART (PLAN.md step 19): "arm scale,wheel scale,correlation time (s),seed" adds noise to the applied commands; the
# clean commands stay the labels. Scale 1 = the split MLP's error on expert frames, measured on 20 validation episodes,
# which also gave the 0.1 s correlation time. White noise at 1 kHz would average out in the joints
_noise = [float(v) for v in os.environ.get("ROBOENV_NOISE", "0,0,0.1,0").split(",")]
ARM_NOISE, WHEEL_NOISE, NOISE_TAU, NOISE_SEED = _noise[0], _noise[1], _noise[2], int(_noise[3])
ARM_NOISE_STD = np.array([0.050, 0.064, 0.035, 0.066, 0.043, 0.077, 0.011])   # N*m
WHEEL_NOISE_STD = 0.24   # rad/s

# Fixed gripper finger configuration (angles in radians, held fixed during simulation):
# [finger_1, finger_tip_1, finger_2, finger_tip_2, finger_3, finger_tip_3]
# Set to None to use initial configuration from config/URDF, or an array of 6 angles.
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Max time (s) allowed for the base to reach its standoff pose before the arm starts reaching.
# Targets behind the robot need a large turn, so this must be generous.
BASE_NAV_TIMEOUT = 25.0
# Max EE distance from the shoulder the arm is asked to reach (m). Targets further away are projected
# onto this sphere, so the arm settles at the closest pose instead of fully stretching (singularity).
ARM_REACH_LIMIT = 0.85
# Base-to-target horizontal standoff (m) as a function of target height: larger for low targets,
# default for mid heights, smaller for high targets (far above the shoulder, which sits ~0.49 m high).
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]
# Stop once the EE is this close to the target (m); time cap set via ROBOENV_MAX_TIME.
# 5 mm, tighter than the course's 1 cm, so the data shows a precise final approach; the policy is judged by the same
# 5 mm (PLAN.md step 0, tolerance update).
REACH_TOLERANCE = 0.005
# Once the base has arrived, stop if the EE error has not improved for this long (s). The expert never
# goes more than 1.75 s without improving on a successful run, so this only ends unreachable targets.
STALL_WINDOW = 2.0
# Drive progress (0..1) at which the arm starts moving to its final pose; 0 = from the start of the drive. Raise it
# if the arm swinging out early unsettles the base (ROBOENV_ARM_PATH_START)
ARM_PATH_START = float(os.environ.get("ROBOENV_ARM_PATH_START", "0.0"))
# ==============================================================================


def main():
    # --------------------------------------------------------------------------
    # Configuration and Simulation Setup
    # --------------------------------------------------------------------------
    conf_file_name = "robotnik_arm.json"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=not headless)

    # Configure realistic ground contact and skid-steer friction dynamics
    sim.SetFloorFriction(1.0)
    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet
    wheel_joint_ids = [
        j for j in range(client.getNumJoints(rid))
        if "wheel" in client.getJointInfo(rid, j)[1].decode()
    ]
    for j in wheel_joint_ids:
        client.changeDynamics(
            rid, j,
            contactStiffness=30000,
            contactDamping=1000,
            anisotropicFriction=[1.0, 0.01, 1.0]
        )
    sim.bot[0].servo_motor_model.set_motor_gains(400, 30)
    # Place the base at its start pose (does nothing for the default origin start)
    set_start_pose(sim, START_POSE)

    # --------------------------------------------------------------------------
    # Gripper Fingers: Fix gripper finger configuration
    # --------------------------------------------------------------------------
    target_finger_angles = configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)

    # Visualize the target reach reference in the PyBullet 3D environment as a red dot
    target_pos_list = TARGET_EE_POS.tolist()
    visual_shape_id = client.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.03,
        rgbaColor=[1.0, 0.0, 0.0, 0.85]
    )
    client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=target_pos_list
    )
    client.addUserDebugPoints(
        pointPositions=[target_pos_list],
        pointColorsRGB=[[1.0, 0.0, 0.0]],
        pointSize=10
    )

    # Active joints from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names_arr = np.expand_dims(np.array(ext_names), axis=0)
    source_names = ["pybullet"]

    # Pinocchio dynamic & kinematic model
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names_arr, source_names, False, 0, cur_dir)

    controlled_frame_name = "robot_j2s7s300_end_effector"
    ee_frame_id = dyn_model.pin_model.getFrameId(controlled_frame_name)
    shoulder_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_link_2")

    init_joint_angles = np.array(sim.GetMotorAngles(0))
    init_base_pos = np.array(sim.GetBasePosition())
    init_base_ori = np.array(sim.GetBaseOrientation())
    init_bearing = quaternion2bearing(init_base_ori[3], init_base_ori[0], init_base_ori[1], init_base_ori[2])

    # Forward kinematics for initial end-effector position
    init_q_all = np.concatenate((init_base_pos, init_base_ori, init_joint_angles))
    init_q_pin = dyn_model.ReoderJoints2PinVec(init_q_all, "pos")
    pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, init_q_pin)
    pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
    init_ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()

    print("==================================================================")
    print(f"Target Cartesian EE Position : {TARGET_EE_POS}")
    print(f"Initial Cartesian EE Position: {np.round(init_ee_pos, 4)}")
    print(f"Initial Cartesian Error (m)  : {np.linalg.norm(TARGET_EE_POS - init_ee_pos):.4f}")
    print("==================================================================")

    # Joint limits
    lower_limits, upper_limits = sim.GetBotJointsLimit()

    time_step = sim.GetTimeStep()
    current_time = 0

    # --------------------------------------------------------------------------
    # Coordinated Motion Planning: Base Navigation & Arm Reach
    # --------------------------------------------------------------------------
    # Standoff distance between robot base center and target, depending on target height.
    standoff = float(np.interp(TARGET_EE_POS[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
    # Parking spot on the line from the base's start position to the target (from the world origin before, which
    # is only right for an origin start; identical results for it)
    target_xy = TARGET_EE_POS[:2]
    to_target = target_xy - init_base_pos[:2]
    target_bearing = np.arctan2(to_target[1], to_target[0]) if np.linalg.norm(to_target) > 1e-4 else 0.0

    # Mobile base goal location facing the target
    base_goal_pos = target_xy - standoff * np.array([np.cos(target_bearing), np.sin(target_bearing)])
    base_goal_bearing = target_bearing

    # Base navigation controller (align -> drive -> orient -> done), see BaseControllerParams
    base_state = BaseNavState()
    base_navigation_done = base_state.done
    base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)

    # Arm controller (differential IK + joint PD), see ArmControllerParams
    arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
    q_init_arm = init_joint_angles[4:11].copy()
    q_des_arm = q_init_arm.copy()
    qd_des_arm = np.zeros(7)
    arm_lower = np.array(lower_limits[4:11])
    arm_upper = np.array(upper_limits[4:11])
    # Final arm pose for the planned parking spot, solved once before the base moves (kinematics only)
    q_final, q_final_error, ik_iters = solve_final_arm_pose(
        dyn_model, np.array([base_goal_pos[0], base_goal_pos[1], init_base_pos[2]]), base_goal_bearing,
        init_joint_angles, TARGET_EE_POS, ee_frame_id, shoulder_frame_id, arm_lower, arm_upper, arm_params,
    )
    print(f"Final arm pose: {np.round(q_final, 3)} (EE error {q_final_error:.4f} m at the planned spot, "
          f"{ik_iters} IK iterations)")
    dist_at_drive_start = None
    # Stop conditions (target reached, time cap, unreachable-target stall), see StopCriteria
    stop_criteria = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_time=max_time, stall_window=STALL_WINDOW)

    # Command interface
    cmd = MotorCommands()

    # Data logging for plotting
    time_all = []
    ee_pos_all = []
    ee_error_all = []
    base_pos_all = []
    q_mes_all = []
    qd_mes_all = []
    q_d_all = []
    qd_d_all = []
    # Added these three for visualisation only - no needed for the model
    base_bearing_all = []
    wheel_cmds_all = []
    torque_arm_all = []
    # The expert's hidden state (not model inputs): labels for the "arrived" flag, ground truth for the RNN
    nav_phase_all = []
    nav_done_all = []
    q_des_arm_all = []
    arm_progress_all = []

    # --------------------------------------------------------------------------
    # Main Simulation Loop
    # --------------------------------------------------------------------------
    stop_message = None
    noise_rng = np.random.default_rng(NOISE_SEED)
    noise_keep = np.exp(-time_step / NOISE_TAU)
    arm_noise, wheel_noise = np.zeros(7), np.zeros(4)
    while True:
        # Measure current states
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))

        # Full kinematic state vector in Pinocchio format
        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")

        # Compute end-effector forward kinematics
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_error = TARGET_EE_POS - ee_pos
        ee_dist_error = np.linalg.norm(ee_error)

        # ----------------------------------------------------------------------
        # 1. Mobile Base Control (Wheel velocities)
        # ----------------------------------------------------------------------
        wheel_cmds = base_controller(
            base_state, base_pos, base_bearing, base_goal_pos, base_goal_bearing, current_time, base_params
        )
        base_navigation_done = base_state.done

        # ----------------------------------------------------------------------
        # 2. Arm Control
        # ----------------------------------------------------------------------
        # While the base travels the arm follows its path to q_final (held still while the base turns to face the
        # goal); once the base has parked, differential IK + joint PD corrects the rest, as in the sequential expert.
        dist_to_goal = np.linalg.norm(base_pos[:2] - base_goal_pos)
        if base_state.phase == "drive" and dist_at_drive_start is None:
            dist_at_drive_start = dist_to_goal
        progress = drive_progress(base_state.phase, dist_to_goal, dist_at_drive_start or 0.0)
        if base_navigation_done:
            torque_arm, q_des_arm, qd_des_arm = arm_controller(
                dyn_model, q_pin, q_mes[4:11], qd_mes[4:11], q_des_arm, q_init_arm, TARGET_EE_POS, ee_pos,
                ee_frame_id, shoulder_frame_id, arm_lower, arm_upper, time_step, arm_params, reach_active=True,
            )
        else:
            q_ref = arm_path_reference(q_init_arm, q_final, progress, ARM_PATH_START)
            torque_arm, q_des_arm, qd_des_arm = arm_path_controller(
                dyn_model, q_pin, q_mes[4:11], qd_mes[4:11], q_des_arm, q_ref, time_step, arm_params,
            )

        # ----------------------------------------------------------------------
        # 3. Gripper / Finger Control (Fingers locked at fixed configuration)
        # ----------------------------------------------------------------------
        torque_fingers = np.zeros(6)

        # ----------------------------------------------------------------------
        # Send Multi-Interface Control Commands
        # ----------------------------------------------------------------------
        # 4 wheels (velocity) + 7 arm joints (torque) + 6 fingers (torque)
        cmd_all = np.concatenate((wheel_cmds, torque_arm, torque_fingers))
        if ARM_NOISE > 0 or WHEEL_NOISE > 0:
            # Correlated noise, std unchanged by the correlation. Wheels only while driving: the base stands still
            # once it has arrived, under the policy too
            spread = np.sqrt(1 - noise_keep ** 2)
            arm_noise = noise_keep * arm_noise + spread * ARM_NOISE * ARM_NOISE_STD * noise_rng.standard_normal(7)
            wheel_noise = noise_keep * wheel_noise + spread * WHEEL_NOISE * WHEEL_NOISE_STD * noise_rng.standard_normal(4)
            cmd_all[4:11] += arm_noise
            if not base_navigation_done:
                cmd_all[:4] += wheel_noise
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        sim.Step(cmd, "torque")

        # Visualization in Pinocchio GUI if active
        if dyn_model.visualizer:
            for index in range(len(sim.bot)):
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)

        # Exit logic with 'q' key in PyBullet GUI window
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # Stop criteria: target reached within fine tolerance, time cap, or unreachable target
        stop_message = check_stop(stop_criteria, current_time, ee_dist_error, base_navigation_done)
        if stop_message:
            print(stop_message)
            break

        # Data logging for plotting
        time_all.append(current_time)
        ee_pos_all.append(ee_pos)
        ee_error_all.append(ee_error)
        base_pos_all.append(base_pos)
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        current_desired_q = np.concatenate((q_mes[:4], q_des_arm, target_finger_angles))
        current_desired_qd = np.concatenate((wheel_cmds, qd_des_arm, np.zeros(6)))
        q_d_all.append(current_desired_q)
        qd_d_all.append(current_desired_qd)
        base_bearing_all.append(base_bearing)
        wheel_cmds_all.append(wheel_cmds)
        torque_arm_all.append(torque_arm)
        # Values after this step's controller updates, i.e. the state that produced this frame's commands
        nav_phase_all.append(BaseNavState.PHASES.index(base_state.phase))
        nav_done_all.append(base_navigation_done)
        q_des_arm_all.append(q_des_arm)
        arm_progress_all.append(progress)

        if not headless:
            time.sleep(0.001)

        current_time += time_step
        if int(current_time / time_step) % 500 == 0:
            print(f"t={current_time:.2f}s | Base pos: {np.round(base_pos[:2], 3)} | EE pos: {np.round(ee_pos, 3)} | EE error: {ee_dist_error:.4f}m | yaw {base_bearing:.2f} phase {base_state.phase} nav_done {base_navigation_done}")

    # Final summary report
    print("==================================================================")
    print(f"Final Base Position          : {np.round(base_pos[:2], 4)}")
    print(f"Final End-Effector Position  : {np.round(ee_pos, 4)}")
    print(f"Target End-Effector Position : {TARGET_EE_POS}")
    print(f"Final Cartesian Error (m)    : {np.linalg.norm(TARGET_EE_POS - ee_pos):.4f}")
    print(f"Fixed Finger Angles          : {np.round(target_finger_angles, 4)}")
    print(f"Final Finger Angles          : {np.round(q_mes[11:17], 4)}")
    print(f"Max Finger Deviation (rad)   : {np.max(np.abs(q_mes[11:17] - target_finger_angles)):.6f}")
    print("==================================================================")

    # Per-frame arrays as float32 (the model trains in float32 anyway): half the file size
    np.savez(
        out_path,
        time=np.array(time_all, dtype=np.float32),
        base_pos=np.array(base_pos_all, dtype=np.float32),
        base_bearing=np.array(base_bearing_all, dtype=np.float32),
        q_mes=np.array(q_mes_all, dtype=np.float32),
        qd_mes=np.array(qd_mes_all, dtype=np.float32),
        wheel_cmds=np.array(wheel_cmds_all, dtype=np.float32),
        torque_arm=np.array(torque_arm_all, dtype=np.float32),
        ee_pos=np.array(ee_pos_all, dtype=np.float32),
        target=TARGET_EE_POS,
        final_error=np.linalg.norm(TARGET_EE_POS - ee_pos),
        stop_message=str(stop_message),
        nav_phase=np.array(nav_phase_all, dtype=np.int8),   # index into nav_phase_names
        nav_phase_names=np.array(BaseNavState.PHASES),
        nav_done=np.array(nav_done_all, dtype=bool),
        q_des_arm=np.array(q_des_arm_all, dtype=np.float32),
        start_pose=np.array([init_base_pos[0], init_base_pos[1], init_bearing]),   # measured at t = 0
        base_goal=np.array([base_goal_pos[0], base_goal_pos[1], base_goal_bearing]),   # parking spot x, y, yaw
        # Concurrent expert only: the final arm pose (the goal network's label), its EE error at the planned spot,
        # and the drive progress that timed the arm path (0 while turning, 1 once the drive is over)
        q_final=q_final.astype(np.float32),
        q_final_error=q_final_error,
        arm_progress=np.array(arm_progress_all, dtype=np.float32),
        arm_path_start=ARM_PATH_START,
    )

    if headless:
        print(f"Headless run finished at t={current_time:.2f}s")
        return

if __name__ == '__main__':
    main()
