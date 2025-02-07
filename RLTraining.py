import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from RLModel import SchedulingEnv


def main():
    # Load datasets
    print("Loading datasets...")
    teachers = pd.read_csv("teachers.csv")
    students = pd.read_csv("students.csv")
    rooms = pd.read_csv("rooms.csv")

    print("Setting up model...")
    env = SchedulingEnv(teachers, students, rooms)
    vec_env = make_vec_env(lambda: env, n_envs=1)

    model = PPO("MultiInputPolicy", vec_env, verbose=1, device="auto",
                learning_rate=0.001,
                gamma=0.99,
                n_steps=1024,
                ent_coef=0.01) # Encourage exploration

    # Adjust time-steps for better results but longer training time
    model.learn(total_timesteps=100000)

    model.save("scheduling_rl_model")


if __name__ == '__main__':
    main()