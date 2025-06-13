import pandas as pd
import numpy as np
from faker import Faker
import random
from config import MAX_TEACHERS, MAX_ROOMS, MAX_STUDENTS, TIME_SLOTS


def generate_test_data():
    fake = Faker()

    np.random.seed(42)

    # Parameters
    number_teachers = random.randint(1, MAX_TEACHERS)
    number_students = random.randint(1, MAX_STUDENTS)
    number_rooms = random.randint(1, MAX_ROOMS)

    # Generate all possible time slots
    all_time_slots = pd.date_range("2025-01-01 08:00", "2025-05-01 17:00", freq="30min")

    # Restrict to the first `TIME_SLOTS` entries
    time_slots = all_time_slots[:TIME_SLOTS]

    # Generate Teachers
    teachers = pd.DataFrame({
        "Teacher_ID": [f"T{i:03d}" for i in range(1, number_teachers + 1)],
        "Instruments": [random.sample(["Piano", "Guitar", "Violin", "Drums"], k=random.randint(1, 3)) for _ in
                        range(number_teachers)],
        "Max_Hours_Per_Week": np.random.randint(5, 15, number_teachers)
    })

    # Generate Students
    students = pd.DataFrame({
        "Student_ID": [f"S{i:03d}" for i in range(1, number_students + 1)],
        "Instrument": [random.choice(["Piano", "Guitar", "Violin", "Drums"]) for _ in range(number_students)],
    })

    # Generate Rooms
    rooms = pd.DataFrame({
        "Room_ID": [f"R{i:02d}" for i in range(1, number_rooms + 1)],
    })

    print("Teachers:\n", teachers.head())
    print("\nStudents:\n", students.head())
    print("\nRooms:\n", rooms.head())

    teachers.to_csv("teachers_test.csv", index=False)
    students.to_csv("students_test.csv", index=False)
    rooms.to_csv("rooms_test.csv", index=False)


def generate_train_data():
    np.random.seed(42)

    # Parameters
    # Generate all possible time slots
    all_time_slots = pd.date_range("2025-01-01 08:00", "2025-05-01 17:00", freq="30min")

    # Restrict to the first `TIME_SLOTS` entries
    time_slots = all_time_slots[:TIME_SLOTS]

    # Generate Teachers
    teachers = pd.DataFrame({
        "Teacher_ID": [f"T{i:03d}" for i in range(1, MAX_TEACHERS + 1)],
        "Instruments": [random.sample(["Piano", "Guitar", "Violin", "Drums"], k=random.randint(1, 3)) for _ in
                        range(MAX_TEACHERS)],
    })

    # Generate Students
    students = pd.DataFrame({
        "Student_ID": [f"S{i:03d}" for i in range(1, MAX_STUDENTS + 1)],
        "Instrument": [random.choice(["Piano", "Guitar", "Violin", "Drums"]) for _ in range(MAX_STUDENTS)],
    })

    # Generate Rooms
    rooms = pd.DataFrame({
        "Room_ID": [f"R{i:02d}" for i in range(1, MAX_ROOMS + 1)],
    })

    times = pd.DataFrame(time_slots, columns=["Time Slot"])


    print("Teachers:\n", teachers.head())
    print("\nStudents:\n", students.head())
    print("\nRooms:\n", rooms.head())
    print("\nTimes:\n", times.head())

    teachers.to_csv("teachers.csv", index=False)
    students.to_csv("students.csv", index=False)
    rooms.to_csv("rooms.csv", index=False)
    times.to_csv("times.csv", index=False)


def main():
    dataset_generation = input("Enter 1 for training data and 2 for test data")

    if dataset_generation == "1":
        generate_train_data()
    elif dataset_generation == "2":
        generate_test_data()
    else:
        print("Enter valid input")


if __name__ == '__main__':
    main()