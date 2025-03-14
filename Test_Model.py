import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from RLModel import SchedulingEnv

def test_model():
    # Load unseen test dataset
    print("Loading test datasets...")
    teachers = pd.read_csv("teachers_test.csv")  # Unseen teachers
    students = pd.read_csv("students_test.csv")  # Unseen students
    rooms = pd.read_csv("rooms_test.csv")        # Unseen rooms

    # Initialize test environment
    test_env = SchedulingEnv(teachers, students, rooms)
    vec_test_env = make_vec_env(lambda: test_env, n_envs=1)

    # Load trained model
    print("Loading trained model...")
    model = PPO.load("scheduling_rl_model")

    # Run inference
    obs = vec_test_env.reset()
    done = False
    schedule = []

    print("Generating schedule...")
    while not done:
        action, _states = model.predict(obs, deterministic=True)  # Predict best action
        obs, rewards, dones, info = vec_test_env.step(action)  # Step through env
        schedule.append(info)  # Store results
        done = dones[0]  # Assuming single environment

    # Print or save schedule
    print("Generated Schedule:", schedule)
    pd.DataFrame(schedule).to_csv("generated_schedule.csv", index=False)
    print("Schedule saved to generated_schedule.csv")

if __name__ == '__main__':
    test_model()