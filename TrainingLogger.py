from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import pandas as pd

# Custom Callback to Log Training Data
class TrainingLoggerCallback(BaseCallback):
    def __init__(self, log_dir="training_logs.csv", verbose=1):
        super(TrainingLoggerCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.training_data = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        # Get rewards and episode end signals
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        # Track episode rewards and lengths
        self.current_episode_reward += np.sum(rewards)
        self.current_episode_length += 1

        if np.any(dones):  # If any environment finished an episode
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0

        # Collect relevant training metrics
        metrics = {
            "timesteps": self.num_timesteps,
            "reward_mean": np.mean(self.episode_rewards[-10:]),  # Last 10 episodes
            "episode_length_mean": np.mean(self.episode_lengths[-10:]),  # Last 10 episodes
            "value_loss": self.model.logger.name_to_value.get("train/value_loss", np.nan),
            "policy_loss": self.model.logger.name_to_value.get("train/policy_gradient_loss", np.nan),
            "explained_variance": self.model.logger.name_to_value.get("train/explained_variance", np.nan),
            "entropy_loss": self.model.logger.name_to_value.get("train/entropy_loss", np.nan),
        }

        self.training_data.append(metrics)
        return True  # Continue training

    def _on_training_end(self):
        # Save collected data to CSV
        df = pd.DataFrame(self.training_data)
        df.to_csv(self.log_dir, index=False)
        print(f"Training log saved to {self.log_dir}")