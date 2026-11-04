"""Does the mast camera change the physics? (PLAN.md step 29) Same commands with and without rendering, every state
compared bit for bit. Run from concurrent_arm/: python check_camera_physics.py
"""
import os, sys
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
sys.path.insert(0, "../robotnik_arm_task")
import pybullet as p
from simulation_and_control import pb, MotorCommands
from mast_camera import MastCameraParams, detect_target


def run(with_camera, steps=4000):
    sim = pb.SimInterface("robotnik_arm.json", conf_file_path_ext=os.getcwd(), use_gui=False)
    client = sim.pybullet_client; rid = sim.bot[0].bot_pybullet
    sim.SetFloorFriction(1.0)
    for j in range(client.getNumJoints(rid)):
        if "wheel" in client.getJointInfo(rid, j)[1].decode():
            client.changeDynamics(rid, j, contactStiffness=30000, contactDamping=1000, anisotropicFriction=[1.0, 0.01, 1.0])
    sim.bot[0].servo_motor_model.set_motor_gains(400, 30)
    vs = client.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 0.85])
    client.createMultiBody(baseMass=0, baseVisualShapeIndex=vs, basePosition=[1.5, 0.3, 0.6])
    mass = sum(client.getDynamicsInfo(rid, j)[0] for j in range(-1, client.getNumJoints(rid)))
    cmd = MotorCommands(); params = MastCameraParams(); states = []
    rng = np.random.default_rng(0)
    torques = rng.normal(0, 2, (steps, 7))
    for k in range(steps):
        wheels = np.array([4.0, 2.0, 2.0, 4.0]) if k < 2500 else np.array([-3.0, 3.0, 3.0, -3.0])
        cmd.SetControlCmd(np.concatenate((wheels, torques[k], np.zeros(6))), ["velocity"] * 4 + ["torque"] * 13)
        if with_camera and k % 100 == 0:
            detect_target(client, np.array(sim.GetBasePosition()), np.array(sim.GetBaseOrientation()), params, rid)
        sim.Step(cmd, "torque")
        states.append(np.concatenate((sim.GetBasePosition(), sim.GetBaseOrientation(),
                                      sim.GetMotorAngles(0), sim.GetMotorVelocities(0))))
    n_bodies = client.getNumBodies()
    client.disconnect()
    return np.array(states), mass, n_bodies


a, mass_a, nb_a = run(False)
b, mass_b, nb_b = run(True)
print(f"robot mass without / with camera: {mass_a:.6f} / {mass_b:.6f} kg; bodies {nb_a} / {nb_b}")
print(f"states over {len(a)} steps (4 s, base drove {np.linalg.norm(a[-1,:2]):.2f} m): "
      f"bit-identical = {np.array_equal(a, b)}, max abs difference = {np.max(np.abs(a - b)):.3g}")
