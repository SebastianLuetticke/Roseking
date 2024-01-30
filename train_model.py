import gymnasium
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from game_env import GameEnv
import os

GAME_NUM = 100
WIN_RATIO = 0.6
TIMESTEPS_BEFORE_UPDATE = 100000
TRAINING_ITERATIONS = 250

models_dir = "models/new_models"
logdir = "logs/new_logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

def mask_fn(env: gymnasium.Env) -> np.ndarray:
    return env.valid_action_mask()

# Create an environment and the agent.
modeltype = 2
# For masked actions
env = GameEnv(model=modeltype)
env = ActionMasker(env, mask_fn)
model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
env.reset()
# Create an environment for the opponent.
opponent_env = GameEnv(model=modeltype)
opponent_env = ActionMasker(opponent_env, mask_fn)
env.set_opponent_env(opponent_env)
# For unmasked actions (against random player, no test games)
"""
env = GameEnv(model=modeltype)
env.reset()
model = PPO("MlpPolicy", env, verbose=1)
"""

for i in range(1,TRAINING_ITERATIONS):
    model.learn(total_timesteps=TIMESTEPS_BEFORE_UPDATE, log_interval=1, reset_num_timesteps=False, tb_log_name="MaskablePPO")
    model_path = f"{models_dir}/{TIMESTEPS_BEFORE_UPDATE*i}"
    model.save(model_path)

    # Play test games to determine whether the agent has improved.
    wins = 0
    for i in tqdm(range(GAME_NUM), desc="Play Games"):
        obs, info = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            mask = mask_fn(env)
            action, _ = model.predict(obs, action_masks=mask)
            obs, reward, terminated, truncated, info = env.step(action)
            if info["won"] == True:
                wins += 1
    print("WINS: ", wins)
    # If the agent has improved, update the opponent.
    if wins / GAME_NUM >= WIN_RATIO:
        print(f"LOAD MODEL: {model_path}.zip")
        opponent_model = MaskablePPO.load(f"{model_path}.zip", env=opponent_env)
        env.set_opponent_model(opponent_model)
