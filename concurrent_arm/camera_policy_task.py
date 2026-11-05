"""Camera policy run: a trained policy driven by the mast camera's estimate instead of the true target (PLAN.md step 29).

A copy of ../robotnik_arm_task/robotnik_arm_policy_task.py (Matteo's; not edited) with the camera expert's perception
(concurrent_arm/camera_expert.py) in front of the policy:
- search: until the camera sees the target the base turns on the spot (scripted, as in the camera expert) and the arm
  holds its start pose with joint PD + gravity (as the concurrent expert's arm does while the base turns); the policy
  can't run yet, it has no target to put in its inputs;
- from the first clean detection the policy drives everything, with the camera's estimate in place of TARGET_EE_POS in
  make_features; the estimate keeps improving at 10 Hz (mast_camera.TargetEstimator), and the arrival latch's 25 s
  time-out counts from the sighting;
- the stop rule and the saved errors stay the referee's, measured to the true target, exactly as in the true-target
  runs (compare.py), so the two are directly comparable. The 30 s time cap includes the search.
ROBOENV_CAMERA=0 turns the camera off: the script then behaves exactly like robotnik_arm_policy_task.py (checked bit for
bit on saved Route B val runs). ROBOENV_CAMERA_REFINE=0: first clean detection only. ROBOENV_POLICY: the checkpoint.
"""
import os
import sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch + numpy each ship an OpenMP runtime on Windows
import numpy as np
import time

# The shared code (controllers, bc_data, bc_split) lives in ../robotnik_arm_task
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robotnik_arm_task"))

# Headless mode and simulated time cap (s) for CI / direct runs. 30 s is about 2x the longest expert
# run: anything not reached by then counts as a failure.
headless = os.environ.get("ROBOENV_HEADLESS", "").strip().lower() in ("1", "true", "yes")
# max_steps = int(os.environ.get("ROBOENV_MAX_STEPS", "40000"))   # replaced by the time cap, see PLAN.md step 0
max_time = float(os.environ.get("ROBOENV_MAX_TIME", "30"))
out_path = os.environ.get("ROBOENV_OUT", "policy_episode.npz")

import matplotlib
if headless:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as p
import pinocchio as pin
from simulation_and_control import pb, MotorCommands, PinWrapper
from robotnik_arm_control import (
    StopCriteria, check_stop, configure_gripper_fingers, parse_start_pose, quaternion2bearing, set_start_pose,
)

import torch
from bc_data import make_features, decode_outputs, arm_gravity, FrameHistory
from bc_split import ArrivedLatch, SplitPolicy, load_policy
from robotnik_arm_control import ArmControllerParams
from concurrent_control import arm_path_controller
from mast_camera import MastCameraParams, TargetEstimator, make_detector, parse_mount
from camera_expert import SEARCH_MAX_TURN, SEARCH_YAW_RATE, search_wheels, wrap
from robotnik_arm_control import BaseControllerParams


# ==============================================================================
# TARGET SPECIFICATION
# Fixed 3D Cartesian coordinates for the Kinova end-effector in the world frame:
# [X (forward/meters), Y (lateral/meters), Z (height/meters)]
_target_str = os.environ.get("ROBOENV_TARGET", "0.0,1.1,0.5")
TARGET_EE_POS = np.array([float(v) for v in _target_str.split(",")])
# Start pose of the base (x m, y m, yaw rad), e.g. ROBOENV_START=1.0,-0.5,1.57. Default: the origin facing +x,
# which all training data uses (PLAN.md step 0, fix 4).
START_POSE = parse_start_pose(os.environ.get("ROBOENV_START", "0,0,0"))

# Fixed gripper finger configuration (angles in radians, held fixed during simulation):
# [finger_1, finger_tip_1, finger_2, finger_tip_2, finger_3, finger_tip_3]
# Set to None to use initial configuration from config/URDF, or an array of 6 angles.
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Stop once the EE is this close to the target (m); time cap set via ROBOENV_MAX_TIME.
# 5 mm, the same success rule as the expert's (PLAN.md step 0, tolerance update).
REACH_TOLERANCE = 0.005
# Camera (PLAN.md step 29): on by default; 0 = the original policy script's behaviour (true target from t = 0)
CAMERA = os.environ.get("ROBOENV_CAMERA", "1").strip().lower() in ("1", "true", "yes")
CAMERA_REFINE = os.environ.get("ROBOENV_CAMERA_REFINE", "1").strip().lower() in ("1", "true", "yes")
# ==============================================================================



def true_ee_pos(sim, dyn_model, ee_frame_id):
    """End-effector position from the simulator's true state, without the config's sensor noise (PLAN.md step 12).
    Overwrites the forward kinematics in dyn_model.pin_data."""
    bot = sim.bot[0]
    q_pin = dyn_model.ReoderJoints2PinVec(np.concatenate((bot.base_position, bot.base_orientation, bot.motor_angles)), "pos")
    pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
    pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
    return dyn_model.pin_data.oMf[ee_frame_id].translation.copy()

def main():
    # --------------------------------------------------------------------------
    # Configuration and Simulation Setup
    # --------------------------------------------------------------------------
    # ROBOENV_CONFIG picks another config in configs/, e.g. one with sensor noise on (PLAN.md step 12); default as before
    conf_file_name = os.environ.get("ROBOENV_CONFIG", "robotnik_arm.json")
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
    if not CAMERA:   # with the camera on, no debug point: it shouldn't be part of what the robot sees
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

    time_step = sim.GetTimeStep()
    current_time = 0

    # --------------------------------------------------------------------------
    # Learned policy (replaces the expert base navigation + arm controllers)
    # --------------------------------------------------------------------------
    # TODO: load the trained PyTorch model here. SET HERE WHICH MODEL TO LOAD
    # Default checkpoint is found relative to this file, so the script works from any working directory
    # A single MLP or split networks (PLAN.md step 15), whichever the checkpoint holds
    model = load_policy(os.environ.get("ROBOENV_POLICY", os.path.join(cur_dir, "..", "robotnik_arm_task", "checkpoints",
                                                                      "mlp_baseline.pt")))
    split = isinstance(model, SplitPolicy)
    # Split networks: switches control from the base to the arm, same rule as in train.py's offline test
    latch = ArrivedLatch()
    torch.set_num_threads(1)   # one sample per step: extra threads only add overhead
    # Past frames the model was trained with (PLAN.md step 14); does nothing for a single-frame model
    history = FrameHistory(model.config.get("history_k", 0), model.config.get("history_spacing", 1))
    stateful = model.config.get("arch") in ("tcn", "gru")   # these keep their own past between steps (step)

    # Stop conditions (target reached, time cap), see StopCriteria. No stall check: while the base turns in
    # place the EE error can go well over 2 s without improving, so a stall rule could cut off a run that
    # would have succeeded.
    stop_criteria = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_time=max_time, stall_window=0.0)

    # Command interface
    cmd = MotorCommands()

    # Data logging for plotting
    time_all = []
    ee_pos_all = []
    ee_error_all = []
    base_pos_all = []
    q_mes_all = []
    qd_mes_all = []
    # Added these three for visualisation only - no needed for the model
    base_bearing_all = []
    wheel_cmds_all = []
    torque_arm_all = []
    # Split networks only: the base network's p(arrived) and the latched flag, per frame
    p_arrived_all = []
    nav_done_all = []

    # Camera and search (PLAN.md step 29). Without the camera the "estimate" is the true target from t = 0
    cam_params = MastCameraParams()
    if os.environ.get("ROBOENV_CAMERA_MOUNT"):
        cam_params.mount_xyz, cam_params.pitch = parse_mount(os.environ["ROBOENV_CAMERA_MOUNT"])
    # Hand (eye-in-hand) camera instead of the mast (5 Nov, Matteo; mast_camera.MastCameraParams.place); default mast
    cam_params.place = os.environ.get("ROBOENV_CAMERA_PLACE", "mast").strip().lower()
    if cam_params.place not in ("mast", "hand"):
        raise ValueError(f"ROBOENV_CAMERA_PLACE={cam_params.place}: use mast or hand")
    cam_every = max(1, int(round(cam_params.period / time_step)))
    estimator = TargetEstimator(refine=CAMERA_REFINE)
    DETECTOR_NAME, detector = make_detector()   # ROBOENV_DETECTOR: classical (default) or cnn
    searching = CAMERA
    t_found = None if CAMERA else 0.0
    search_turned, last_bearing = 0.0, init_bearing
    base_params = BaseControllerParams(nav_timeout=25.0)
    hold_params = ArmControllerParams(reach_limit=0.85)   # only kp, kd, max_torque, max_joint_vel are used here
    q_init_arm = init_joint_angles[4:11].copy()
    q_des_arm = q_init_arm.copy()
    step = 0
    searching_all, target_est_all = [], []
    cam_time, cam_status, cam_pos = [], [], []   # status: 0 none, 1 clean, 2 hidden by the robot, 3 image edge

    # --------------------------------------------------------------------------
    # Main Simulation Loop
    # --------------------------------------------------------------------------
    stop_message = None
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
        # With sensor noise on (PLAN.md step 12) the policy gets the noisy hand position, but the stop rule, the log
        # and the saved ee_pos use the true one: the noise is on the robot's sensors, not on the referee
        ee_seen = ee_pos
        if sim.bot[0].noise_flag:
            ee_pos = true_ee_pos(sim, dyn_model, ee_frame_id)
            ee_error = TARGET_EE_POS - ee_pos
            ee_dist_error = np.linalg.norm(ee_error)

        # ----------------------------------------------------------------------
        # 0. Perception (camera on): detections -> estimate; the policy only runs once there is one
        # ----------------------------------------------------------------------
        if CAMERA:
            if step % cam_every == 0:
                det = detector(client, base_pos, base_ori, cam_params, rid)
                estimator.update(det, current_time)
                cam_time.append(current_time)
                cam_status.append(0 if det is None else 1 if det.clean else 2 if det.occluded else 3)
                cam_pos.append(np.full(3, np.nan) if det is None else det.position)
                if searching and estimator.found:
                    searching, t_found = False, current_time
                    print(f"t={current_time:.2f}s | Target seen after turning {np.rad2deg(search_turned):.0f} deg, "
                          f"estimate error {np.linalg.norm(estimator.estimate - TARGET_EE_POS)*1000:.1f} mm")
            if searching:
                search_turned += abs(wrap(base_bearing - last_bearing))
            last_bearing = base_bearing
            target_est = estimator.estimate
        else:
            target_est = TARGET_EE_POS

        if searching:
            # Scripted until the target is seen: turn on the spot, arm held at its start pose (joint PD + gravity)
            wheel_cmds = search_wheels(SEARCH_YAW_RATE, base_params)
            torque_arm, q_des_arm, _ = arm_path_controller(
                dyn_model, q_pin, q_mes[4:11], qd_mes[4:11], q_des_arm, q_init_arm, time_step, hold_params)
            if search_turned > SEARCH_MAX_TURN:
                stop_message = f"Target not found: turned {np.rad2deg(search_turned):.0f} deg by t={current_time:.2f}s"
            p_arrived, arrived = 0.0, False
        # ----------------------------------------------------------------------
        # 1 + 2. Learned policy: wheel velocities and arm torques from the measurements and the target (estimate)
        # ----------------------------------------------------------------------
        else:
            x = make_features(base_pos, base_bearing, q_mes[4:11], qd_mes[4:11], target_est, ee_seen)
            if split:
                # Base network: wheels + p(arrived). Once the flag is latched the base stops and the arm network,
                # which gets the flag as an input, reaches (before it holds), like the expert
                wheels, p_arrived = model.step_base(x)
                was_arrived = latch.arrived
                arrived = latch.update(p_arrived, current_time - t_found)   # time-out counted from the sighting
                if arrived:
                    wheels = np.zeros_like(wheels)   # 4 commands or (v, w): zero either way means stopped
                    if not was_arrived:
                        print(f"Arrived flag latched at t={current_time:.2f}s ({latch.source})")
                y = np.concatenate([wheels, model.step_arm(x, float(arrived))])
            else:
                y = model.step(x) if stateful else model.predict(history.add(x))
            # Gravity-residual models (PLAN.md step 12) predict torque minus gravity: add this frame's gravity back
            tau_g = arm_gravity(q_mes)[0] if model.config.get("gravity_residual") else None
            wheel_cmds, torque_arm = decode_outputs(y, tau_g, model.config.get("vw_outputs", False))
            # Safety clips to the expert's own ranges (PLAN.md, "Parked"), so one extreme output can't fling the robot away
            wheel_cmds = np.clip(wheel_cmds, -3.0, 3.0)  # rad/s, the expert's largest command (turning in place at 1 rad/s)
            torque_arm = np.clip(torque_arm, -30.0, 30.0)  # same torque limit the expert uses
        # ----------------------------------------------------------------------
        # 3. Gripper / Finger Control (Fingers locked at fixed configuration)
        # ----------------------------------------------------------------------
        torque_fingers = np.zeros(6)

        # ----------------------------------------------------------------------
        # Send Multi-Interface Control Commands
        # ----------------------------------------------------------------------
        # 4 wheels (velocity) + 7 arm joints (torque) + 6 fingers (torque)
        cmd_all = np.concatenate((wheel_cmds, torque_arm, torque_fingers))
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

        # Stop criteria: target reached within fine tolerance, or time cap
        # The policy has no navigation phase, so the reached check is always active
        if stop_message is None:
            stop_message = check_stop(stop_criteria, current_time, ee_dist_error, True)
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
        base_bearing_all.append(base_bearing)
        wheel_cmds_all.append(wheel_cmds)
        torque_arm_all.append(torque_arm)
        searching_all.append(searching)
        target_est_all.append(np.full(3, np.nan) if target_est is None else target_est)
        if split:
            p_arrived_all.append(p_arrived)
            nav_done_all.append(arrived)

        if not headless:
            time.sleep(0.001)

        current_time += time_step
        step += 1
        if int(current_time / time_step) % 500 == 0:
            print(f"t={current_time:.2f}s | Base pos: {np.round(base_pos[:2], 3)} | EE pos: {np.round(ee_pos, 3)} | EE error: {ee_dist_error:.4f}m | yaw {base_bearing:.2f}")

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

    # Split networks: the flag under the expert's name (nav_done), so the two can be compared frame by frame, plus
    # when and why it latched ("network" / "timeout"; "never" with arrived_at nan)
    split_fields = {}
    if split:
        split_fields = dict(
            p_arrived=np.array(p_arrived_all, dtype=np.float32),
            nav_done=np.array(nav_done_all, dtype=bool),
            arrived_source=latch.source or "never",
            arrived_at=time_all[int(np.argmax(nav_done_all))] if any(nav_done_all) else np.nan,
        )

    # Camera runs only (PLAN.md step 29): the estimate per frame (nan while searching) and the camera's detections
    camera_fields = {}
    if CAMERA:
        est = estimator.estimate
        camera_fields = dict(
            searching=np.array(searching_all, dtype=bool),
            target_est=np.array(target_est_all, dtype=np.float32),
            target_est_final=np.full(3, np.nan) if est is None else est,
            est_error=np.nan if est is None else np.linalg.norm(est - TARGET_EE_POS),
            first_seen_time=np.nan if t_found is None else t_found,
            search_turned=search_turned,
            n_detections_used=estimator.n_used,
            camera_refine=CAMERA_REFINE,
            detector=DETECTOR_NAME,
            camera_mount=np.array(list(cam_params.mount_xyz) + [cam_params.pitch]),
            camera_place=cam_params.place,
            cam_time=np.array(cam_time, dtype=np.float32),
            cam_status=np.array(cam_status, dtype=np.int8),
            cam_pos=np.array(cam_pos, dtype=np.float32).reshape(-1, 3),
        )

    # Same file format as the expert's episodes (float32 per-frame arrays)
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
        start_pose=np.array([init_base_pos[0], init_base_pos[1], init_bearing]),   # measured at t = 0
        **split_fields,
        **camera_fields,
    )

    if headless:
        print(f"Headless run finished at t={current_time:.2f}s")
        return
    '''
    # --------------------------------------------------------------------------
    # Plotting Performance
    # --------------------------------------------------------------------------
    ee_pos_all = np.array(ee_pos_all)
    ee_error_all = np.array(ee_error_all)
    time_all = np.array(time_all)

    # 1. Cartesian end-effector coordinates tracking
    plt.figure(figsize=(10, 8))
    coords = ['X', 'Y', 'Z']
    for idx in range(3):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_all, ee_pos_all[:, idx], label=f'Actual EE {coords[idx]}', color='b')
        plt.axhline(y=TARGET_EE_POS[idx], color='r', linestyle='--', label=f'Target EE {coords[idx]}')
        plt.ylabel(f'{coords[idx]} Position (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('End-Effector Cartesian Position Reaching')
    plt.tight_layout()
    plt.show()

    # 2. Cartesian distance error over time
    plt.figure(figsize=(8, 4))
    ee_dist_norm = np.linalg.norm(ee_error_all, axis=1)
    plt.plot(time_all, ee_dist_norm, label='EE Euclidean Error (m)', color='g')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Cartesian End-Effector Reaching Error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Arm joint position tracking
    plt.figure(figsize=(10, 8))
    for i in range(7):
        plt.subplot(4, 2, i + 1)
        plt.plot(time_all, [q[4 + i] for q in q_mes_all], label=f'Measured J{i+1}')
        plt.ylabel(f'J{i+1} (rad)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('Arm Joint Position Profiles')
    plt.tight_layout()
    plt.show()
    '''

if __name__ == '__main__':
    main()
