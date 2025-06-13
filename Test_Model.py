import pandas as pd
from stable_baselines3 import PPO

import config
from RLModel import SchedulingEnv


def test_model():
    print("Loading test datasets...")
    teachers = pd.read_csv("teachers.csv")
    students = pd.read_csv("students.csv")
    rooms = pd.read_csv("rooms.csv")
    times = pd.read_csv("times.csv")

    print("Initializing test environment...")
    test_env = SchedulingEnv(teachers, students, rooms, times)

    print("Loading trained model...")
    model = PPO.load("scheduling_rl_model")

    obs, _ = test_env.reset()
    schedule = []

    print("Generating schedule...")
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=False)

        teacher_idx, room_idx, time_slot = action

        if teacher_idx >= config.MAX_TEACHERS or room_idx >= config.MAX_ROOMS:
            print("⚠️ Out of bounds, skipping action\n"
                  f"Teacher Index: {teacher_idx}"
                  f"Room Index: {room_idx}")
            continue

        decoded = test_env.decode_action(action)
        print("Decoded action:", decoded)

        obs, reward, done, truncated, info = test_env.step(action)

        print(f"\n--- Step Log ---")
        print(f"Action taken: {action}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")

        if info.get("new_lesson") is not None and "error" not in info:
            schedule.append(info["new_lesson"])
            print(f"Scheduled: {info['new_lesson']}")

        if "error" in info:
            print("Error from env:", info["error"])

    print("\nGenerated Schedule:")
    for entry in schedule:
        print(entry)

    if schedule:
        print(schedule)
        pd.DataFrame(schedule).to_csv("generated_schedule.csv", index=False)
        print("Schedule saved to generated_schedule.csv")
    else:
        print("No valid schedule was generated. Current schedule:")
        print(schedule)


if __name__ == '__main__':
    test_model()
