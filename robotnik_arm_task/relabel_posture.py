"""Offline relabel of DART episodes with the posture part of the expert's arm torque removed (PLAN.md C18e).

The expert tracks a reference q_des that it integrates over time and the policy never sees. With 7 joints and a
3D position task, part of its torque only changes the posture (the arm's null space, no force at the hand); near the
target that part pulls the posture back to wherever q_des drifted, which the policy can't predict. After the latch,
each label is split at the measured pose with the orthogonal projector P = J+ J (J the 3x7 position Jacobian; P
also projects torques onto the J^T F ones): the hand part P (tau - g) is kept, the posture part is replaced by the
expert's own damping on the measured joint velocity, -kd (I - P) qd, which the policy sees. Before the latch the arm
holds its pose and the labels are copied.

Done on the labels, not inside the expert's integrator: replaying the expert offline with the null-space part of
q_des - q removed each step leaves an offset that never unwinds (offline, the arm doesn't follow the new q_des), and
puts ~15 N*m of spurious hand torque into clean runs at the target.

    pixi run -e ml python relabel_posture.py dart/a60 dart/a60b      # -> dart/a60_np, dart/a60b_np
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pinocchio as pin

import expert_relabel as er


# Joints 1, 3 and 5: the ones that drift together on low targets (PLAN.md C18e, 4 Nov); 0 in the expert's usual
# reaching posture
SPRING_JOINTS = [0, 2, 4]


def remove_posture(d, start, kp_ref=0.0):
    """Arm torques of a saved run d with the posture part replaced by damping from frame start on.

    kp_ref > 0 (PLAN.md C18e, 4 Nov) adds a spring pulling SPRING_JOINTS towards 0 inside the posture part:
    -kp_ref q on those joints, projected with (I - P) like the damping, so it moves only the spare joints, never the
    hand. Damping alone stops the posture being pulled to the expert's hidden q_des but nothing brings a drifted arm
    back; the spring does, and depends only on the measured joints, which the policy sees. kp_ref = 0: as before.
    """
    torque = d["torque_arm"].astype(float).copy()
    for i in range(start, len(torque)):
        q_pin = er._q_pin(d["base_pos"][i], d["base_bearing"][i], d["q_mes"][i])
        J = pin.computeFrameJacobian(er._MODEL, er._DATA, q_pin, er._EE_FRAME, pin.LOCAL_WORLD_ALIGNED)[:3, 6:13]
        P = np.linalg.pinv(J) @ J
        g = pin.computeGeneralizedGravity(er._MODEL, er._DATA, q_pin)[6:13]
        spring = np.zeros(7)
        spring[SPRING_JOINTS] = -kp_ref * d["q_mes"][i, 4:11][SPRING_JOINTS]
        torque[i] = g + P @ (torque[i] - g) + (np.eye(7) - P) @ (spring - er._PARAMS.kd * d["qd_mes"][i, 4:11])
    return np.clip(torque, -er._PARAMS.max_torque, er._PARAMS.max_torque)


def one(args):
    src, dst = args
    d = dict(np.load(src, allow_pickle=True))
    latched = np.flatnonzero(d["nav_done"] > 0.5)
    if len(latched):
        d["torque_arm"] = remove_posture(d, latched[0]).astype(d["torque_arm"].dtype)
    np.savez(dst, **d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", help="DART episode folders (run_batch.py --noise output)")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    jobs = []
    for folder in args.folders:
        out = Path(folder.rstrip("/") + "_np")
        out.mkdir(exist_ok=True)
        jobs += [(str(f), str(out / f.name)) for f in sorted(Path(folder).glob("ep_*.npz")) if not (out / f.name).exists()]
    with Pool(args.workers) as p:
        p.map(one, jobs, chunksize=1)
    print(f"relabelled {len(jobs)} episodes")
