import pandas as pd
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from RLModel import SchedulingEnv
from TrainingLogger import TrainingLoggerCallback


# Load datasets globally to avoid reloading in each trial
teachers = pd.read_csv("teachers.csv")
students = pd.read_csv("students.csv")
rooms = pd.read_csv("rooms.csv")

# Define the objective function for Bayesian Optimization
def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)  # Updated from suggest_uniform
    n_steps = trial.suggest_int("n_steps", 512, 4096, step=512)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)  # Updated from suggest_uniform
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)  # Updated from suggest_loguniform

    # Create environment
    env = SchedulingEnv(teachers, students, rooms)
    vec_env = make_vec_env(lambda: env, n_envs=1)

    # Initialize PPO with suggested hyperparameters
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=0,  # Reduce verbosity to speed up optimization
        device="auto",
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        clip_range=clip_range,
        ent_coef=ent_coef
    )

    # Train for a small number of timesteps to evaluate performance
    log_callback = TrainingLoggerCallback(log_dir="hyper_paramater_training_logs.csv")
    model.learn(total_timesteps=50000, callback=log_callback)

    # Evaluate model performance
    logs = pd.read_csv("hyper_paramater_training_logs.csv")
    final_reward = logs["reward_mean"].iloc[-1]
    final_entropy = logs["entropy_loss"].iloc[-1]
    final_explained_variance = logs["explained_variance"].iloc[-1]
    value_loss_std = logs["value_loss"].std()  # Lower is better
    policy_loss_std = logs["policy_loss"].std()  # Lower is better

    # Weighted score (maximize reward, minimize entropy loss & variance)
    score = (
            final_reward  # Maximize reward
            - abs(final_entropy) * 10  # Push entropy loss closer to 0
            - value_loss_std * 5  # Reduce value loss variance
            - policy_loss_std * 5  # Reduce policy loss variance
            + final_explained_variance * 10  # Improve explained variance
    )

    return score  # Optuna will maximize this value

def main():
    # Optimize hyperparameters
    study = optuna.create_study(direction="maximize")  # Maximize reward
    study.optimize(objective, n_trials=20)  # Run 20 trials

    # Get best hyperparameters
    best_params = study.best_params
    print("Best hyperparameters found:", best_params)

    # Train final model with optimal hyperparameters
    env = SchedulingEnv(teachers, students, rooms)
    vec_env = make_vec_env(lambda: env, n_envs=1)

    model = PPO("MultiInputPolicy", vec_env, verbose=1, device="auto", **best_params)

    log_callback = TrainingLoggerCallback(log_dir="hyper_paramater_training_logs.csv")
    model.learn(total_timesteps=100000, callback=log_callback)

    # Save final model
    model.save("scheduling_rl_model")

if __name__ == '__main__':
    main()