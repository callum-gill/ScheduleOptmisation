import pandas as pd
import matplotlib.pyplot as plt

# Load the log file
data = pd.read_csv("training_logs.csv")

plt.figure(figsize=(15, 5))

# Plot mean reward over time
plt.subplot(1, 3, 1)
plt.plot(data["timesteps"], data["reward_mean"], label="Episode Reward Mean", color="blue")
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.title("Training Progress - Reward")
plt.legend()

# Plot mean episode length over time
plt.subplot(1, 3, 2)
plt.plot(data["timesteps"], data["episode_length_mean"], label="Episode Length Mean", color="orange")
plt.xlabel("Timesteps")
plt.ylabel("Mean Episode Length")
plt.title("Training Progress - Episode Length")
plt.legend()

# Plot entropy loss over time
plt.subplot(1, 3, 3)
if "entropy_loss" in data.columns:  # Ensure entropy_loss exists in the CSV
    plt.plot(data["timesteps"], data["entropy_loss"], label="Entropy Loss", color="red")
    plt.xlabel("Timesteps")
    plt.ylabel("Entropy Loss")
    plt.title("Entropy Loss Over Time")
    plt.legend()
else:
    print("Warning: 'entropy_loss' column not found in the CSV.")

plt.tight_layout()
plt.show()
