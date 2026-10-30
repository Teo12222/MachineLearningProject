import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch + numpy each ship an OpenMP runtime on Windows
import numpy as np
import time

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
from bc_data import make_features, decode_outputs, arm_gravity
from bc_split import ArrivedLatch, SplitPolicy, load_policy


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
    model = load_policy(os.environ.get("ROBOENV_POLICY", os.path.join(cur_dir, "checkpoints", "mlp_baseline.pt")))
    split = isinstance(model, SplitPolicy)
    # Split networks: switches control from the base to the arm, same rule as in train.py's offline test
    latch = ArrivedLatch()
    torch.set_num_threads(1)   # one sample per step: extra threads only add overhead

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

        # ----------------------------------------------------------------------
        # 1 + 2. Learned policy: wheel velocities and arm torques from the measurements and the target
        # ----------------------------------------------------------------------
        x = make_features(base_pos, base_bearing, q_mes[4:11], qd_mes[4:11], TARGET_EE_POS, ee_pos)
        if split:
            # Base network: wheels + p(arrived). Once the flag is latched the base stops and the arm network,
            # which gets the flag as an input, reaches (before it holds), like the expert
            wheels, p_arrived = model.predict_base(x)
            was_arrived = latch.arrived
            arrived = latch.update(p_arrived, current_time)
            if arrived:
                wheels = np.zeros(4)
                if not was_arrived:
                    print(f"Arrived flag latched at t={current_time:.2f}s ({latch.source})")
            y = np.concatenate([wheels, model.predict_arm(x, float(arrived))])
        else:
            y = model.predict(x)
        # Gravity-residual models (PLAN.md step 12) predict torque minus gravity: add this frame's gravity back
        tau_g = arm_gravity(q_mes)[0] if model.config.get("gravity_residual") else None
        wheel_cmds, torque_arm = decode_outputs(y, tau_g)
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
        if split:
            p_arrived_all.append(p_arrived)
            nav_done_all.append(arrived)

        if not headless:
            time.sleep(0.001)

        current_time += time_step
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
