import pandas as pd
import numpy as np
from faker import Faker
import random


def generate_test_data():
    fake = Faker()

    np.random.seed(42)

    # Parameters
    number_teachers = 25
    number_students = 500
    number_rooms = 20
    number_lessons = 20
    lesson_durations = [30, 60]
    time_slots = pd.date_range("2025-01-01 8:00", "2025-05-01 17:00", freq="30min")

    # Generate Teachers
    teachers = pd.DataFrame({
        "Teacher_ID": [f"T{i:03d}" for i in range(1, number_teachers + 1)],
        "Name": [fake.name() for _ in range(number_teachers)],
        "Instruments": [random.sample(["Piano", "Guitar", "Violin", "Drums"], k=random.randint(1, 3)) for _ in
                        range(number_teachers)],
        "Max_Hours_Per_Week": np.random.randint(5, 15, number_teachers)
    })

    # Generate Students
    students = pd.DataFrame({
        "Student_ID": [f"S{i:03d}" for i in range(1, number_students + 1)],
        "Name": [fake.name() for _ in range(number_students)],
        "Preferred_Times": [random.sample(time_slots.strftime("%H:%M").tolist(), k=5) for _ in range(number_students)],
        "Instrument": [random.choice(["Piano", "Guitar", "Violin", "Drums"]) for _ in range(number_students)],
        "Skill_Level": [random.choice(["Beginner", "Intermediate", "Advanced"]) for _ in range(number_students)],
        "Lessons_Per_Week": np.random.randint(1, 4, number_students)
    })

    # Generate Rooms
    rooms = pd.DataFrame({
        "Room_ID": [f"R{i:02d}" for i in range(1, number_rooms + 1)],
        "Equipment": [random.sample(["Piano", "Guitar", "Violin", "Drums"], k=random.randint(1, 4)) for _ in
                      range(number_rooms)],
        "Available_Slots": [random.sample(time_slots.strftime("%H:%M").tolist(), k=30) for _ in range(number_rooms)]
    })

    # Generate Lessons (Not needed for model only to visualise data)
    lessons = []
    generated_lessons = 0
    while generated_lessons < number_lessons:
        teacher = teachers.sample(1).iloc[0]
        student = students.sample(1).iloc[0]
        room = rooms.sample(1).iloc[0]

        # Ensure compatibility
        if student["Instrument"] in teacher["Instruments"] and student["Instrument"] in room["Equipment"]:
            generated_lessons += 1
            lesson_time = random.choice(room["Available_Slots"])
            lessons.append({
                "Lesson_ID": f"L{generated_lessons:04d}",
                "Teacher_ID": teacher["Teacher_ID"],
                "Student_ID": student["Student_ID"],
                "Room_ID": room["Room_ID"],
                "Time_Slot": lesson_time,
                "Duration": random.choice(lesson_durations),
                "Instrument": student["Instrument"],
                "Skill_Level": student["Skill_Level"]
            })

    lessons_df = pd.DataFrame(lessons)

    print("Teachers:\n", teachers.head())
    print("\nStudents:\n", students.head())
    print("\nRooms:\n", rooms.head())
    print("\nLessons:\n", lessons_df.head())

    teachers.to_csv("teachers_test.csv", index=False)
    students.to_csv("students_test.csv", index=False)
    rooms.to_csv("rooms_test.csv", index=False)


def generate_train_data():
    fake = Faker()

    np.random.seed(42)

    # Parameters
    number_teachers = 25
    number_students = 1000
    number_rooms = 25
    number_lessons = 20
    lesson_durations = [30, 60]
    time_slots = pd.date_range("2025-01-01 8:00", "2025-05-01 17:00", freq="30min")

    # Generate Teachers
    teachers = pd.DataFrame({
        "Teacher_ID": [f"T{i:03d}" for i in range(1, number_teachers + 1)],
        "Name": [fake.name() for _ in range(number_teachers)],
        "Instruments": [random.sample(["Piano", "Guitar", "Violin", "Drums"], k=random.randint(1, 3)) for _ in
                        range(number_teachers)],
        "Max_Hours_Per_Week": np.random.randint(5, 15, number_teachers)
    })

    # Generate Students
    students = pd.DataFrame({
        "Student_ID": [f"S{i:03d}" for i in range(1, number_students + 1)],
        "Name": [fake.name() for _ in range(number_students)],
        "Preferred_Times": [random.sample(time_slots.strftime("%H:%M").tolist(), k=5) for _ in range(number_students)],
        "Instrument": [random.choice(["Piano", "Guitar", "Violin", "Drums"]) for _ in range(number_students)],
        "Skill_Level": [random.choice(["Beginner", "Intermediate", "Advanced"]) for _ in range(number_students)],
        "Lessons_Per_Week": np.random.randint(1, 4, number_students)
    })

    # Generate Rooms
    rooms = pd.DataFrame({
        "Room_ID": [f"R{i:02d}" for i in range(1, number_rooms + 1)],
        "Equipment": [random.sample(["Piano", "Guitar", "Violin", "Drums"], k=random.randint(1, 4)) for _ in
                      range(number_rooms)],
        "Available_Slots": [random.sample(time_slots.strftime("%H:%M").tolist(), k=30) for _ in range(number_rooms)]
    })

    # Generate Lessons (Not needed for model only to visualise data)
    lessons = []
    generated_lessons = 0
    while generated_lessons < number_lessons:
        teacher = teachers.sample(1).iloc[0]
        student = students.sample(1).iloc[0]
        room = rooms.sample(1).iloc[0]

        # Ensure compatibility
        if student["Instrument"] in teacher["Instruments"] and student["Instrument"] in room["Equipment"]:
            generated_lessons += 1
            lesson_time = random.choice(room["Available_Slots"])
            lessons.append({
                "Lesson_ID": f"L{generated_lessons:04d}",
                "Teacher_ID": teacher["Teacher_ID"],
                "Student_ID": student["Student_ID"],
                "Room_ID": room["Room_ID"],
                "Time_Slot": lesson_time,
                "Duration": random.choice(lesson_durations),
                "Instrument": student["Instrument"],
                "Skill_Level": student["Skill_Level"]
            })

    lessons_df = pd.DataFrame(lessons)

    print("Teachers:\n", teachers.head())
    print("\nStudents:\n", students.head())
    print("\nRooms:\n", rooms.head())
    print("\nLessons:\n", lessons_df.head())

    teachers.to_csv("teachers.csv", index=False)
    students.to_csv("students.csv", index=False)
    rooms.to_csv("rooms.csv", index=False)


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