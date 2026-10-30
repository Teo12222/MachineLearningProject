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
