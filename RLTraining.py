import pandas as pd
from stable_baselines3 import PPO
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

    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        device="auto",
        learning_rate=1e-4,
        gamma=0.99,
        n_steps=2048,
        clip_range=0.2,
        ent_coef=0.05
    )

    # Train model with logging
    log_callback = TrainingLoggerCallback(log_dir="training_logs.csv")
    model.learn(total_timesteps=100000, callback=log_callback)

    # Save model
    model.save("scheduling_rl_model")


if __name__ == '__main__':
    main()
