"""The expert's arm controller run offline along saved states: DAgger labels (PLAN.md step 21).

Gives the torques the expert would have commanded at each frame of any saved run (e.g. a policy episode), without
the simulator: the robot model is rebuilt from the URDF and robotnik_arm_control.arm_controller is stepped along the
saved states, carrying its joint reference q_des from frame to frame as the expert does.

Approximation: the base is taken as upright (only its yaw is saved), so the labels are off by the base's small tilt;
on expert episodes they reproduce the saved torques to ~0.01 N*m median. Check with:
    pixi run python expert_relabel.py
"""
import numpy as np
import pinocchio as pin
from pathlib import Path

from robotnik_arm_control import ArmControllerParams, arm_controller
from robotnik_arm_task import ARM_REACH_LIMIT

# Robot model, built once when the module is imported. Pinocchio orders the joints its own way: look up where each
# arm/finger joint sits by name
_MODEL = pin.buildModelFromUrdf(
    str(Path(__file__).resolve().parent / "models" / "summit_xl_arm_description" / "summit_xl_complete_pin.urdf"),
    pin.JointModelFreeFlyer())
_DATA = _MODEL.createData()
_JOINTS = ([f"robot_j2s7s300_joint_{i}" for i in range(1, 8)]
           + [f"robot_j2s7s300_joint_finger{s}" for s in ("_1", "_tip_1", "_2", "_tip_2", "_3", "_tip_3")])
_IDX_Q = [_MODEL.joints[_MODEL.getJointId(n)].idx_q for n in _JOINTS]   # arm (7) then fingers (6), as in q_mes[4:17]
_EE_FRAME = _MODEL.getFrameId("robot_j2s7s300_end_effector")
_SHOULDER_FRAME = _MODEL.getFrameId("robot_j2s7s300_link_2")
# The URDF's limits of the arm joints: the expert reads the simulator's, which mark the continuous joints as
# limitless; here those get +-1e6, which the controller treats the same way (its limit terms never act)
_ARM_LOWER = _MODEL.lowerPositionLimit[_IDX_Q[:7]]
_ARM_UPPER = _MODEL.upperPositionLimit[_IDX_Q[:7]]
_PARAMS = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)   # same settings as the expert
_DT = 0.001   # s, the simulation step


class _Dyn:
    """The two attributes of the expert's PinWrapper that arm_controller uses."""
    pin_model, pin_data = _MODEL, _DATA


def _q_pin(base_pos, base_bearing, q_mes):
    """Pinocchio configuration of one frame: base upright at its position and yaw, arm and fingers as measured."""
    q = pin.neutral(_MODEL)
    q[:3] = base_pos
    q[3:7] = [0.0, 0.0, np.sin(base_bearing / 2), np.cos(base_bearing / 2)]   # quaternion x, y, z, w
    q[_IDX_Q] = q_mes[4:17]
    return q


def relabel_arm(d, start, stop, q_des):
    """The expert's arm torques (N*m, gravity included) for frames start..stop-1 of a saved run d.

    d holds the per-frame arrays base_pos, base_bearing, q_mes, qd_mes, ee_pos and the target. The expert is
    reaching throughout; q_des is its joint reference just before frame start (the arm's start pose when the
    reach begins, as the expert holds that pose until it arrives). Returns an array (stop - start, 7).
    """
    q_des = np.asarray(q_des, dtype=float)
    target = np.asarray(d["target"], dtype=float)
    torques = np.empty((stop - start, 7))
    for k, i in enumerate(range(start, stop)):
        q_pin = _q_pin(d["base_pos"][i], d["base_bearing"][i], d["q_mes"][i])
        pin.forwardKinematics(_MODEL, _DATA, q_pin)
        pin.updateFramePlacements(_MODEL, _DATA)
        torques[k], q_des, _ = arm_controller(
            _Dyn, q_pin, d["q_mes"][i, 4:11].astype(float), d["qd_mes"][i, 4:11].astype(float), q_des, None,
            target, d["ee_pos"][i].astype(float), _EE_FRAME, _SHOULDER_FRAME, _ARM_LOWER, _ARM_UPPER, _DT, _PARAMS,
            reach_active=True)
    return torques


DRIVE_SPEED = 0.05     # m/s: forward speed of the policy's wheel commands above which its first turn is over
WHEEL_RADIUS = 0.11    # m, BaseControllerParams.wheel_radius
POS_TOL = 0.08         # m, BaseControllerParams.pos_tol: the concurrent expert's drive ends this close to its spot


def relabel_arm_drive(d, latch, q_final, goal_xy):
    """Route B drive-phase DAgger (PLAN.md C19 / step 28, 4 Nov): the concurrent expert's arm torques for frames
    0..latch-1 of a policy run d, i.e. while the base turns and drives, as concurrent_arm/concurrent_expert.py
    commands them (early IK off, the setting of data_route_b), along the policy's own states.

    q_final, goal_xy: the expert's final arm pose and planned parking spot for this target (stored in its episode).
    Progress as the expert's drive_progress: 0 while the base turns in place (here: until the policy's wheel
    commands first drive forward faster than DRIVE_SPEED, as the policy has no phase), then
    1 - distance to the spot / that distance when the drive started, and 1 from when the base is within POS_TOL of
    the spot (the expert's drive ends there; its final turn holds q_final). The arm follows arm_path_reference /
    arm_path_controller from its start pose, unchanged, plus the sequential expert's lead limit (joint reference at
    most max_q_lead from the measured angle; inactive on the expert's own runs, but the policy's arm can be off the
    path, PLAN.md C19 Route A). From the latch on, dagger.label_run's Cartesian reach from q_final takes over, as in
    the expert with early IK off. Returns (latch, 7) torques.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "concurrent_arm"))
    from concurrent_control import arm_path_controller, arm_path_reference
    q_start = d["q_mes"][0, 4:11].astype(float)
    q_des = q_start.copy()
    speed = WHEEL_RADIUS * np.asarray(d["wheel_cmds"], dtype=float).mean(axis=1)
    torques = np.empty((latch, 7))
    d0, done = None, False
    for i in range(latch):
        dist = float(np.linalg.norm(d["base_pos"][i, :2] - goal_xy))
        if d0 is None and speed[i] > DRIVE_SPEED:
            d0 = dist                                     # the drive starts: the expert's dist_at_drive_start
        done = done or (d0 is not None and dist < POS_TOL)
        progress = 0.0 if d0 is None else 1.0 if done else float(np.clip(1.0 - dist / max(d0, 1e-6), 0.0, 1.0))
        q_ref = arm_path_reference(q_start, q_final, progress)
        q_arm = d["q_mes"][i, 4:11].astype(float)
        q_des = np.clip(q_des, q_arm - _PARAMS.max_q_lead, q_arm + _PARAMS.max_q_lead)
        q_pin = _q_pin(d["base_pos"][i], d["base_bearing"][i], d["q_mes"][i])
        torques[i], q_des, _ = arm_path_controller(
            _Dyn, q_pin, q_arm, d["qd_mes"][i, 4:11].astype(float), q_des, q_ref, _DT, _PARAMS)
    return torques


def relabel_arm_postpark(d, latch, stop, q_final):
    """Sequential pre-shaping DAgger (PLAN.md step 28, 5 Nov): the arm torques of concurrent_expert.py with
    ROBOENV_ARM_SEQUENTIAL=1 for frames 0..stop-1 of a policy run d whose latch (first nav_done frame) is latch.

    Before the latch: the start pose held by arm_path_controller (path fraction 0), as the expert does while the base
    turns and drives. From the latch: arm_path_reference(start pose, q_final, u), u = steps since the latch * dt /
    blend_duration(start pose, q_final), with the 0.15 rad lead limit, both as in the expert; from the first frame with
    u >= 1 the Cartesian reach (arm_controller) carries on from the path's joint reference, i.e. q_final, as the expert
    switches. The switch is on time since the latch, which the policy can't see; but it is where the path ends (arm
    at q_final and at rest), which it can. Returns (torques (stop, 7), switch frame; stop if the path doesn't end).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "concurrent_arm"))
    from concurrent_control import arm_path_controller, arm_path_reference, blend_duration
    q_start = d["q_mes"][0, 4:11].astype(float)
    q_final = np.asarray(q_final, dtype=float)
    path_time = blend_duration(q_start, q_final, _PARAMS.max_joint_vel)
    target = np.asarray(d["target"], dtype=float)
    q_des = q_start.copy()
    torques = np.empty((stop, 7))
    switch = stop
    for i in range(stop):
        q_arm = d["q_mes"][i, 4:11].astype(float)
        qd_arm = d["qd_mes"][i, 4:11].astype(float)
        q_pin = _q_pin(d["base_pos"][i], d["base_bearing"][i], d["q_mes"][i])
        u = 0.0 if i < latch else (i - latch) * _DT / path_time
        if u < 1.0:
            q_ref = arm_path_reference(q_start, q_final, u)
            q_des = np.clip(q_des, q_arm - _PARAMS.max_q_lead, q_arm + _PARAMS.max_q_lead)
            torques[i], q_des, _ = arm_path_controller(_Dyn, q_pin, q_arm, qd_arm, q_des, q_ref, _DT, _PARAMS)
        else:
            switch = min(switch, i)
            pin.forwardKinematics(_MODEL, _DATA, q_pin)
            pin.updateFramePlacements(_MODEL, _DATA)
            torques[i], q_des, _ = arm_controller(
                _Dyn, q_pin, q_arm, qd_arm, q_des, None, target, d["ee_pos"][i].astype(float), _EE_FRAME,
                _SHOULDER_FRAME, _ARM_LOWER, _ARM_UPPER, _DT, _PARAMS, reach_active=True)
    return torques, switch


if __name__ == "__main__":
    # Check: on expert episodes, relabelling the 3 s after arrival (starting from the expert's own reference) must
    # give back the torques it saved
    for ep in (2, 12, 68, 416, 954):
        d = np.load(Path(__file__).resolve().parent / "data_main" / f"ep_{ep:04d}.npz")
        k = int(np.argmax(d["nav_done"]))
        n = min(3000, len(d["time"]) - k)
        err = np.abs(relabel_arm(d, k, k + n, d["q_des_arm"][k - 1]) - d["torque_arm"][k:k + n])
        print(f"episode {ep}: relabelled vs saved torques over {n / 1000:.1f} s: median {np.median(err):.4f}, "
              f"99% {np.percentile(err, 99):.3f}, max {err.max():.3f} N*m")
