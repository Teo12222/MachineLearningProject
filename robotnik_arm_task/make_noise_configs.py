"""Copies of configs/robotnik_arm.json with the simulator's sensor noise on, for the robustness runs (PLAN.md step 12).

The simulator adds Gaussian noise to what GetMotorAngles / GetMotorVelocities / GetBasePosition / GetBaseOrientation
return when noise_flag is set; the *_cov values are standard deviations (rad, rad/s, m, rad), not variances. One level
sets every channel to the same sigma. The shared robotnik_arm.json is never touched: a run picks a copy with
ROBOENV_CONFIG=robotnik_arm_noise<mm>.json.

    python make_noise_configs.py            # -> configs/robotnik_arm_noise1.json, _noise5.json, _noise10.json
"""
import json
from pathlib import Path

CONFIGS = Path(__file__).resolve().parent / "configs"
LEVELS_MM = (1, 5, 10)   # sigma in mm / mrad (and mrad/s)

if __name__ == "__main__":
    base = json.loads((CONFIGS / "robotnik_arm.json").read_text())
    for level in LEVELS_MM:
        conf = json.loads(json.dumps(base))
        bot = conf["robot_pybullet"]
        bot["noise_flag"] = [1]
        bot["robot_noise"] = [{key: level / 1000 for key in bot["robot_noise"][0]}]
        out = CONFIGS / f"robotnik_arm_noise{level}.json"
        out.write_text(json.dumps(conf, indent=4))
        print(f"{out.name}: noise_flag on, every channel sigma {level / 1000}")
