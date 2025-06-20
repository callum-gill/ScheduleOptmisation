import pandas as pd
import matplotlib.pyplot as plt


def main(filename):
    # Load the log file
    data = pd.read_csv(filename)

    plt.figure(figsize=(15, 10))

    # Plot mean reward over time
    plt.subplot(2, 3, 1)
    plt.plot(data["timesteps"], data["reward_mean"], label="Episode Reward Mean", color="blue")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("Training Progress - Reward")
    plt.legend()

    # Plot mean episode length over time
    plt.subplot(2, 3, 2)
    plt.plot(data["timesteps"], data["episode_length_mean"], label="Episode Length Mean", color="orange")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Length")
    plt.title("Training Progress - Episode Length")
    plt.legend()

    # Plot entropy loss over time
    plt.subplot(2, 3, 3)
    if "entropy_loss" in data.columns:
        plt.plot(data["timesteps"], data["entropy_loss"], label="Entropy Loss", color="red")
        plt.xlabel("Timesteps")
        plt.ylabel("Entropy Loss")
        plt.title("Entropy Loss Over Time")
        plt.legend()

    # Plot value loss over time
    plt.subplot(2, 3, 4)
    if "value_loss" in data.columns:
        plt.plot(data["timesteps"], data["value_loss"], label="Value Loss", color="purple")
        plt.xlabel("Timesteps")
        plt.ylabel("Value Loss")
        plt.title("Value Loss Over Time")
        plt.legend()

    # Plot policy loss over time
    plt.subplot(2, 3, 5)
    if "policy_loss" in data.columns:
        plt.plot(data["timesteps"], data["policy_loss"], label="Policy Loss", color="green")
        plt.xlabel("Timesteps")
        plt.ylabel("Policy Loss")
        plt.title("Policy Loss Over Time")
        plt.legend()

    # Plot explained variance over time
    plt.subplot(2, 3, 6)
    if "explained_variance" in data.columns:
        plt.plot(data["timesteps"], data["explained_variance"], label="Explained Variance", color="brown")
        plt.xlabel("Timesteps")
        plt.ylabel("Explained Variance")
        plt.title("Explained Variance Over Time")
        plt.legend()

    plt.tight_layout()
    plt.savefig("training_plots.png")

if __name__ == '__main__':
    fileNameOption = input("Enter 1 for training_logs.csv and 2 for hyper_paramater_training_logs.csv")
    if fileNameOption == "1":
        main("training_logs.csv")
    elif fileNameOption == "2":
        main("hyper_paramater_training_logs.csv")
    else:
        print("Invalid file option")
