"""Camera expert: the concurrent expert, but the target is found and located with a mast camera (PLAN.md step 29).

Same simulation, base controller, arm path, Cartesian IK and stop rules as concurrent_expert.py (PLAN.md step 28).
What changes is that the robot no longer knows TARGET_EE_POS; it only uses its camera estimate:
- search: if the target is not in view, the base turns on the spot (counter-clockwise) until the camera sees it;
  after a full turn plus margin without a sighting the run stops ("Target not found");
- from the first clean detection: parking spot, final arm pose (q_final) and the base's time-out are computed from
  the estimate, then the concurrent expert runs as before (align -> drive with the arm moving -> orient + Cartesian IK);
- the camera keeps looking (10 Hz) and the estimate keeps improving as the base gets closer (weighted towards close
  views, mast_camera.TargetEstimator). The parking spot follows the estimate; q_final is solved again while the base
  is still turning or driving if the estimate moved more than RESOLVE_SHIFT; the Cartesian IK always uses the latest
  estimate. Detections touching the robot's own body in the image (the arm near the end of the reach) are skipped.
- the stop rule uses the error to the estimate (all the robot knows), so "Target reached" means within 5 mm of the
  estimate; the saved final_error is the true error (evaluation only).

Settings (environment): ROBOENV_CAMERA_MOUNT=x,y,z,pitch_deg (default mast_camera.MastCameraParams),
ROBOENV_CAMERA_REFINE=0 keeps the first clean detection only (the "seen once" baseline); the concurrent expert's own
switches (ROBOENV_ARM_EARLY_IK, ROBOENV_ARM_RESOLVE, ...) work as there.
Saved episodes have the concurrent expert's fields plus the estimate per frame and the camera's detections.
"""
import numpy as np
import time
import os

# Headless mode and simulated time cap (s) for CI / direct runs
headless = os.environ.get("ROBOENV_HEADLESS", "").strip().lower() in ("1", "true", "yes")
max_time = float(os.environ.get("ROBOENV_MAX_TIME", "40"))
out_path = os.environ.get("ROBOENV_OUT", "episode.npz")

import matplotlib
if headless:
    matplotlib.use("Agg")
import pybullet as p
import pinocchio as pin
from simulation_and_control import pb, MotorCommands, PinWrapper
# Puts ../robotnik_arm_task on the path too: the shared controllers live with the sequential expert
from concurrent_control import (
    arm_path_controller, arm_path_reference, blend_duration, drive_progress, solve_final_arm_pose,
)
from robotnik_arm_control import (
    ArmControllerParams, BaseControllerParams, BaseNavState, StopCriteria,
    arm_controller, base_controller, check_stop, configure_gripper_fingers, parse_start_pose, quaternion2bearing,
    set_start_pose,
)
from mast_camera import MastCameraParams, TargetEstimator, detect_target, parse_mount


# ==============================================================================
# TARGET SPECIFICATION
# True target position (world frame): only used to place the red sphere and to report the true error.
_target_str = os.environ.get("ROBOENV_TARGET", "0.0,1.1,0.5")
TARGET_EE_POS = np.array([float(v) for v in _target_str.split(",")])
START_POSE = parse_start_pose(os.environ.get("ROBOENV_START", "0,0,0"))
# DART noise, as in concurrent_expert.py (PLAN.md step 19)
_noise = [float(v) for v in os.environ.get("ROBOENV_NOISE", "0,0,0.1,0").split(",")]
ARM_NOISE, WHEEL_NOISE, NOISE_TAU, NOISE_SEED = _noise[0], _noise[1], _noise[2], int(_noise[3])
ARM_NOISE_STD = np.array([0.050, 0.064, 0.035, 0.066, 0.043, 0.077, 0.011])   # N*m
WHEEL_NOISE_STD = 0.24   # rad/s

FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Max time (s) for the base to reach its standoff pose, counted from the first sighting of the target
BASE_NAV_TIMEOUT = 25.0
ARM_REACH_LIMIT = 0.85
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]
REACH_TOLERANCE = 0.005
STALL_WINDOW = 2.0
ARM_PATH_START = float(os.environ.get("ROBOENV_ARM_PATH_START", "0.0"))
ARM_RESOLVE = os.environ.get("ROBOENV_ARM_RESOLVE", "0").strip().lower() in ("1", "true", "yes")
ARM_EARLY_IK = os.environ.get("ROBOENV_ARM_EARLY_IK", "1").strip().lower() in ("1", "true", "yes")
CARTESIAN_KP = float(os.environ.get("ROBOENV_CARTESIAN_KP", "1.5"))

# Camera and search
CAMERA_REFINE = os.environ.get("ROBOENV_CAMERA_REFINE", "1").strip().lower() in ("1", "true", "yes")
SEARCH_YAW_RATE = 0.8                 # rad/s commanded while searching (4.6 deg per camera frame, FOV 80 deg)
SEARCH_MAX_TURN = 2 * np.pi + 0.5     # rad turned without a sighting before giving up
# Solve q_final again (while the base turns or drives) when the estimate moved this far since the last solve (m)
RESOLVE_SHIFT = 0.01
# ==============================================================================


def search_wheels(yaw_rate, params):
    """Wheel commands for turning on the spot at yaw_rate (same skid-steer model as base_controller)."""
    lw = (-yaw_rate * params.wheel_base_width / 2) / params.wheel_radius
    rw = (yaw_rate * params.wheel_base_width / 2) / params.wheel_radius
    return np.array([rw, lw, lw, rw])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    # --------------------------------------------------------------------------
    # Configuration and Simulation Setup
    # --------------------------------------------------------------------------
    conf_file_name = "robotnik_arm.json"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=not headless)

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
    set_start_pose(sim, START_POSE)

    target_finger_angles = configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)

    # The red target sphere (what the camera looks for). No debug point: it would be drawn into the camera image
    visual_shape_id = client.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1.0, 0.0, 0.0, 0.85])
    client.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=TARGET_EE_POS.tolist())

    ext_names = sim.getNameActiveJoints()
    ext_names_arr = np.expand_dims(np.array(ext_names), axis=0)
    source_names = ["pybullet"]
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names_arr, source_names, False, 0, cur_dir)

    controlled_frame_name = "robot_j2s7s300_end_effector"
    ee_frame_id = dyn_model.pin_model.getFrameId(controlled_frame_name)
    shoulder_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_link_2")

    init_joint_angles = np.array(sim.GetMotorAngles(0))
    init_base_pos = np.array(sim.GetBasePosition())
    init_base_ori = np.array(sim.GetBaseOrientation())
    init_bearing = quaternion2bearing(init_base_ori[3], init_base_ori[0], init_base_ori[1], init_base_ori[2])

    print("==================================================================")
    print(f"Target Cartesian EE Position : {TARGET_EE_POS} (unknown to the robot)")
    print("==================================================================")

    lower_limits, upper_limits = sim.GetBotJointsLimit()
    time_step = sim.GetTimeStep()
    current_time = 0

    # --------------------------------------------------------------------------
    # Camera, search and planning state
    # --------------------------------------------------------------------------
    cam_params = MastCameraParams()
    if os.environ.get("ROBOENV_CAMERA_MOUNT"):
        cam_params.mount_xyz, cam_params.pitch = parse_mount(os.environ["ROBOENV_CAMERA_MOUNT"])
    cam_every = max(1, int(round(cam_params.period / time_step)))
    estimator = TargetEstimator(refine=CAMERA_REFINE)
    searching = True
    search_turned, last_bearing = 0.0, init_bearing
    t_found, approach_from = None, None
    base_goal_pos, base_goal_bearing = None, None

    base_state = BaseNavState()
    base_navigation_done = False
    base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)

    arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT, cartesian_kp=CARTESIAN_KP)
    solve_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
    q_init_arm = init_joint_angles[4:11].copy()
    q_des_arm = q_init_arm.copy()
    qd_des_arm = np.zeros(7)
    arm_lower = np.array(lower_limits[4:11])
    arm_upper = np.array(upper_limits[4:11])
    q_final, q_final_error = q_init_arm.copy(), np.nan
    est_at_solve, n_solves = None, 0
    dist_at_drive_start = None
    q_final_parked, q_final_parked_error = np.full(7, np.nan), np.nan
    resolve_time, blend_time, blend_from = None, 0.0, None
    stop_criteria = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_time=max_time, stall_window=STALL_WINDOW)

    def plan_from_estimate(target_est, base_z):
        """Parking spot on the line from where the target was first seen to the estimate, facing the estimate."""
        standoff = float(np.interp(target_est[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
        to_target = target_est[:2] - approach_from
        bearing = np.arctan2(to_target[1], to_target[0]) if np.linalg.norm(to_target) > 1e-4 else 0.0
        goal = target_est[:2] - standoff * np.array([np.cos(bearing), np.sin(bearing)])
        return goal, bearing

    def solve_q_final(target_est, q_motors_start, base_z):
        q, err, iters = solve_final_arm_pose(
            dyn_model, np.array([base_goal_pos[0], base_goal_pos[1], base_z]), base_goal_bearing,
            q_motors_start, target_est, ee_frame_id, shoulder_frame_id, arm_lower, arm_upper, solve_params,
        )
        return q, err, iters

    cmd = MotorCommands()

    time_all, ee_pos_all, ee_error_all, base_pos_all = [], [], [], []
    q_mes_all, qd_mes_all, q_d_all, qd_d_all = [], [], [], []
    base_bearing_all, wheel_cmds_all, torque_arm_all = [], [], []
    nav_phase_all, nav_done_all, q_des_arm_all, arm_progress_all, arm_mode_all = [], [], [], [], []
    # Camera expert only
    searching_all, target_est_all = [], []
    cam_time, cam_status, cam_pos, cam_npx, cam_dist = [], [], [], [], []   # status: 0 none, 1 clean, 2 hidden, 3 edge

    # --------------------------------------------------------------------------
    # Main Simulation Loop
    # --------------------------------------------------------------------------
    stop_message = None
    noise_rng = np.random.default_rng(NOISE_SEED)
    noise_keep = np.exp(-time_step / NOISE_TAU)
    arm_noise, wheel_noise = np.zeros(7), np.zeros(4)
    step = 0
    while True:
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))

        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_error = TARGET_EE_POS - ee_pos   # true error, for evaluation and logging only

        # ----------------------------------------------------------------------
        # 0. Perception: camera -> estimate -> parking spot (and q_final while the arm is still on its path)
        # ----------------------------------------------------------------------
        if step % cam_every == 0:
            det = detect_target(client, base_pos, base_ori, cam_params, rid)
            changed = estimator.update(det, current_time)
            cam_time.append(current_time)
            cam_status.append(0 if det is None else 1 if det.clean else 2 if det.occluded else 3)
            cam_pos.append(np.full(3, np.nan) if det is None else det.position)
            cam_npx.append(0 if det is None else det.n_pixels)
            cam_dist.append(np.nan if det is None else det.distance)
            if changed:
                target_est = estimator.estimate
                if searching:
                    searching = False
                    t_found, approach_from = current_time, base_pos[:2].copy()
                    print(f"t={current_time:.2f}s | Target seen after turning {np.rad2deg(search_turned):.0f} deg, "
                          f"estimate {np.round(target_est, 3)} (error {np.linalg.norm(target_est - TARGET_EE_POS)*1000:.1f} mm)")
                base_goal_pos, base_goal_bearing = plan_from_estimate(target_est, base_pos[2])
                still_on_path = base_state.phase in ("align", "drive") and not base_navigation_done
                if still_on_path and (est_at_solve is None or np.linalg.norm(target_est - est_at_solve) > RESOLVE_SHIFT):
                    q_start = q_mes.copy()
                    q_start[4:11] = q_final if est_at_solve is not None else q_init_arm
                    q_final, q_final_error, _ = solve_q_final(target_est, q_start, init_base_pos[2])
                    est_at_solve, n_solves = target_est.copy(), n_solves + 1
                    pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
                    pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        target_est = estimator.estimate
        # The robot only knows its estimate: the stop rule and the arm use the error to it
        ee_dist_error = np.linalg.norm(target_est - ee_pos) if target_est is not None else np.inf

        # ----------------------------------------------------------------------
        # 1. Mobile Base Control: search turn until the target is seen, then the usual navigation
        # ----------------------------------------------------------------------
        if searching:
            search_turned += abs(wrap(base_bearing - last_bearing))
        last_bearing = base_bearing
        if searching:
            wheel_cmds = search_wheels(SEARCH_YAW_RATE, base_params)
            if search_turned > SEARCH_MAX_TURN:
                stop_message = f"Target not found: turned {np.rad2deg(search_turned):.0f} deg by t={current_time:.2f}s"
        else:
            # The time-out counts from the first sighting
            wheel_cmds = base_controller(
                base_state, base_pos, base_bearing, base_goal_pos, base_goal_bearing, current_time - t_found,
                base_params,
            )
            base_navigation_done = base_state.done

        # ----------------------------------------------------------------------
        # 2. Arm Control (concurrent expert; holds the start pose while searching)
        # ----------------------------------------------------------------------
        if searching:
            progress = 0.0
        else:
            dist_to_goal = np.linalg.norm(base_pos[:2] - base_goal_pos)
            if base_state.phase == "drive" and dist_at_drive_start is None:
                dist_at_drive_start = dist_to_goal
            progress = drive_progress(base_state.phase, dist_to_goal, dist_at_drive_start or 0.0)
        if ARM_RESOLVE and not searching and resolve_time is None and base_state.phase not in ("align", "drive"):
            q_motors = q_mes.copy()
            q_motors[4:11] = q_des_arm
            q_final_parked, q_final_parked_error, _ = solve_final_arm_pose(
                dyn_model, base_pos, base_goal_bearing, q_motors, target_est, ee_frame_id, shoulder_frame_id,
                arm_lower, arm_upper, solve_params,
            )
            pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
            pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
            blend_from, resolve_time = q_des_arm.copy(), current_time
            blend_time = blend_duration(blend_from, q_final_parked, arm_params.max_joint_vel)
        blending = resolve_time is not None and current_time - resolve_time < blend_time
        ik_active = not searching and (base_navigation_done or (ARM_EARLY_IK and base_state.phase not in ("align", "drive")))
        if ik_active and not blending:
            arm_mode = 2
            torque_arm, q_des_arm, qd_des_arm = arm_controller(
                dyn_model, q_pin, q_mes[4:11], qd_mes[4:11], q_des_arm, q_init_arm, target_est, ee_pos,
                ee_frame_id, shoulder_frame_id, arm_lower, arm_upper, time_step, arm_params, reach_active=True,
            )
        else:
            if blending:
                arm_mode = 1
                q_ref = arm_path_reference(blend_from, q_final_parked, (current_time - resolve_time) / blend_time)
            else:
                arm_mode = 0
                q_ref = arm_path_reference(q_init_arm, q_final, progress, ARM_PATH_START)
            torque_arm, q_des_arm, qd_des_arm = arm_path_controller(
                dyn_model, q_pin, q_mes[4:11], qd_mes[4:11], q_des_arm, q_ref, time_step, arm_params,
            )

        torque_fingers = np.zeros(6)
        cmd_all = np.concatenate((wheel_cmds, torque_arm, torque_fingers))
        if ARM_NOISE > 0 or WHEEL_NOISE > 0:
            spread = np.sqrt(1 - noise_keep ** 2)
            arm_noise = noise_keep * arm_noise + spread * ARM_NOISE * ARM_NOISE_STD * noise_rng.standard_normal(7)
            wheel_noise = noise_keep * wheel_noise + spread * WHEEL_NOISE * WHEEL_NOISE_STD * noise_rng.standard_normal(4)
            cmd_all[4:11] += arm_noise
            if not base_navigation_done:
                cmd_all[:4] += wheel_noise
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        sim.Step(cmd, "torque")

        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        if stop_message is None:
            stop_message = check_stop(stop_criteria, current_time, ee_dist_error, base_navigation_done)
        if stop_message:
            print(stop_message)
            break

        time_all.append(current_time)
        ee_pos_all.append(ee_pos)
        ee_error_all.append(ee_error)
        base_pos_all.append(base_pos)
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(np.concatenate((q_mes[:4], q_des_arm, target_finger_angles)))
        qd_d_all.append(np.concatenate((wheel_cmds, qd_des_arm, np.zeros(6))))
        base_bearing_all.append(base_bearing)
        wheel_cmds_all.append(wheel_cmds)
        torque_arm_all.append(torque_arm)
        # While searching the phase is logged as "align" (0); `searching` tells them apart
        nav_phase_all.append(BaseNavState.PHASES.index(base_state.phase))
        nav_done_all.append(base_navigation_done)
        q_des_arm_all.append(q_des_arm)
        arm_progress_all.append(progress)
        arm_mode_all.append(arm_mode)
        searching_all.append(searching)
        target_est_all.append(target_est if target_est is not None else np.full(3, np.nan))

        if not headless:
            time.sleep(0.001)

        current_time += time_step
        step += 1
        if int(current_time / time_step) % 500 == 0:
            est_txt = "searching" if searching else f"est err {np.linalg.norm(target_est - TARGET_EE_POS)*1000:.1f}mm"
            print(f"t={current_time:.2f}s | Base pos: {np.round(base_pos[:2], 3)} | EE true error: "
                  f"{np.linalg.norm(ee_error):.4f}m | {est_txt} | phase {base_state.phase} nav_done {base_navigation_done}")

    true_error = np.linalg.norm(TARGET_EE_POS - ee_pos)
    est_error = np.linalg.norm(target_est - TARGET_EE_POS) if target_est is not None else np.nan
    print("==================================================================")
    print(f"Final Base Position          : {np.round(base_pos[:2], 4)}")
    print(f"Final End-Effector Position  : {np.round(ee_pos, 4)}")
    print(f"Target End-Effector Position : {TARGET_EE_POS}")
    print(f"Final Cartesian Error (m)    : {true_error:.4f} (true)")
    if target_est is not None:
        print(f"Estimated Target Position    : {np.round(target_est, 4)} (first seen t={t_found:.2f}s, "
              f"{estimator.n_used} clean detections, q_final solved {n_solves}x)")
        print(f"Target Estimation Error (m)  : {est_error:.4f}")
    else:
        print("Estimated Target Position    : never seen")
    print("==================================================================")

    nan3 = np.full(3, np.nan)
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
        final_error=true_error,
        stop_message=str(stop_message),
        nav_phase=np.array(nav_phase_all, dtype=np.int8),
        nav_phase_names=np.array(BaseNavState.PHASES),
        nav_done=np.array(nav_done_all, dtype=bool),
        q_des_arm=np.array(q_des_arm_all, dtype=np.float32),
        start_pose=np.array([init_base_pos[0], init_base_pos[1], init_bearing]),
        base_goal=np.array([np.nan] * 3 if base_goal_pos is None else [base_goal_pos[0], base_goal_pos[1], base_goal_bearing]),
        q_final=q_final.astype(np.float32),
        q_final_error=q_final_error,
        arm_progress=np.array(arm_progress_all, dtype=np.float32),
        arm_path_start=ARM_PATH_START,
        arm_mode=np.array(arm_mode_all, dtype=np.int8),
        arm_resolve=ARM_RESOLVE,
        arm_early_ik=ARM_EARLY_IK,
        cartesian_kp=CARTESIAN_KP,
        q_final_parked=q_final_parked.astype(np.float32),
        q_final_parked_error=q_final_parked_error,
        resolve_time=np.nan if resolve_time is None else resolve_time,
        # Camera expert only. target_est: the robot's estimate per frame (nan while searching); final_error_est: the
        # EE error to the final estimate (what the stop rule saw); est_error: final estimate vs the true target
        searching=np.array(searching_all, dtype=bool),
        target_est=np.array(target_est_all, dtype=np.float32),
        target_est_final=nan3 if target_est is None else target_est,
        est_error=est_error,
        final_error_est=np.nan if target_est is None else np.linalg.norm(target_est - ee_pos),
        first_seen_time=np.nan if t_found is None else t_found,
        search_turned=search_turned,
        n_detections_used=estimator.n_used,
        n_q_final_solves=n_solves,
        camera_refine=CAMERA_REFINE,
        camera_mount=np.array(list(cam_params.mount_xyz) + [cam_params.pitch]),
        cam_time=np.array(cam_time, dtype=np.float32),
        cam_status=np.array(cam_status, dtype=np.int8),
        cam_pos=np.array(cam_pos, dtype=np.float32).reshape(-1, 3),
        cam_n_pixels=np.array(cam_npx, dtype=np.int32),
        cam_distance=np.array(cam_dist, dtype=np.float32),
    )

    if headless:
        print(f"Headless run finished at t={current_time:.2f}s")
        return


if __name__ == '__main__':
    main()
