import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from RLModel import SchedulingEnv


def test_model():
    print("Loading test datasets...")
    teachers = pd.read_csv("teachers_test.csv")
    students = pd.read_csv("students_test.csv")
    rooms = pd.read_csv("rooms_test.csv")

    print("Initializing test environment...")

    def make_env():
        return SchedulingEnv(teachers, students, rooms)

    test_env = DummyVecEnv([make_env])

    print("Loading trained model...")
    model = PPO.load("scheduling_rl_model")

    obs = test_env.reset()
    schedule = []

    print("Generating schedule...")
    done = [False]
    while not all(done):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)

        print(f"Action taken: {action}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")

        if isinstance(info, list) and "schedule" in info[0]:
            schedule.extend(info[0]["schedule"])

    # Save schedule
    print("Generated Schedule:", schedule)
    if schedule:
        pd.DataFrame(schedule).to_csv("generated_schedule.csv", index=False)
        print("Schedule saved to generated_schedule.csv")
    else:
        print("No valid schedule was generated.")


if __name__ == '__main__':
    test_model()
