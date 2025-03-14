import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from RLModel import SchedulingEnv
from TrainingLogger import TrainingLoggerCallback


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
                learning_rate=0.009532797821928342,
                gamma=0.9714174390463426,
                n_steps=4096,
                clip_range=0.8973883616872791,
                ent_coef=0.00010841606232106418)  # Encourage exploration

    # Train model with logging
    log_callback = TrainingLoggerCallback(log_dir="training_logs.csv")
    model.learn(total_timesteps=100000, callback=log_callback)

    # Save model
    model.save("scheduling_rl_model")


if __name__ == '__main__':
    main()
