import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config import MAX_TEACHERS, MAX_ROOMS, MAX_STUDENTS, TIME_SLOTS

class SchedulingEnv(gym.Env):
    def __init__(self, teachers_df, students_df, rooms_df):
        super(SchedulingEnv, self).__init__()

        self.teachers = teachers_df.reset_index(drop=True)
        self.students = students_df.reset_index(drop=True)
        self.rooms = rooms_df.reset_index(drop=True)

        # Mappings
        self.teacher_ids = self.teachers["Teacher_ID"].tolist()
        self.student_ids = self.students["Student_ID"].tolist()
        self.room_ids = self.rooms["Room_ID"].tolist()

        self.teacher_mapping = {tid: idx for idx, tid in enumerate(self.teacher_ids)}
        self.student_mapping = {sid: idx for idx, sid in enumerate(self.student_ids)}
        self.room_mapping = {rid: idx for idx, rid in enumerate(self.room_ids)}

        # Fixed-size action space
        self.action_space = spaces.MultiDiscrete([
            MAX_TEACHERS,
            MAX_STUDENTS,
            MAX_ROOMS,
            TIME_SLOTS
        ])

        # Fixed-size observation space
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self.schedule = []

    def reset(self, seed=None, options=None):
        self.schedule = []
        return self.get_obs(), {}

    def get_obs(self):
        teacher = self.teachers.sample(1).iloc[0]
        student = self.students.sample(1).iloc[0]
        room = self.rooms.sample(1).iloc[0]

        student_skill = (
            1 if student["Skill_Level"] == "Beginner"
            else 2 if student["Skill_Level"] == "Intermediate"
            else 3
        ) / 3

        obs = np.array([
            self.teacher_mapping[teacher["Teacher_ID"]] / MAX_TEACHERS,
            self.student_mapping[student["Student_ID"]] / MAX_STUDENTS,
            self.room_mapping[room["Room_ID"]] / MAX_ROOMS,
            np.random.randint(0, TIME_SLOTS) / TIME_SLOTS,
            teacher["Max_Hours_Per_Week"] / 15,
            float(student["Instrument"] in teacher["Instruments"]),
            float(student["Instrument"] in room["Equipment"]),
            student_skill
        ], dtype=np.float32)

        return obs

    def step(self, action):
        teacher_idx, student_idx, room_idx, time_slot = action

        # Check bounds and validity
        if (teacher_idx >= len(self.teachers) or
            student_idx >= len(self.students) or
            room_idx >= len(self.rooms)):
            reward = -1.0
            done = False
            return self.get_obs(), reward, done, False, {}

        teacher_id = self.teacher_ids[teacher_idx]
        student_id = self.student_ids[student_idx]
        room_id = self.room_ids[room_idx]

        teacher = self.teachers.iloc[teacher_idx]
        student = self.students.iloc[student_idx]
        room = self.rooms.iloc[room_idx]

        valid = self._is_valid_action(teacher_id, student_id, room_id, time_slot)

        reward = 0
        if valid:
            self.schedule.append((teacher_id, student_id, room_id, time_slot))
            reward += 10
            if student["Lessons_Per_Week"] > 2:
                reward += 5
            if student["Instrument"] in teacher["Instruments"] and student["Instrument"] in room["Equipment"]:
                reward += 10
        else:
            reward -= 50

        # Normalize reward between 0 and 1
        normalized_reward = (reward + 50) / 75  # from range [-50,25] to [0,1]

        done = len(self.schedule) >= len(self.students)

        return self.get_obs(), normalized_reward, done, False, {"schedule": self.schedule.copy()}

    def _is_valid_action(self, teacher_id, student_id, room_id, time_slot):
        teacher = self.teachers[self.teachers["Teacher_ID"] == teacher_id].iloc[0]
        student = self.students[self.students["Student_ID"] == student_id].iloc[0]
        room = self.rooms[self.rooms["Room_ID"] == room_id].iloc[0]

        if student["Instrument"] not in teacher["Instruments"]:
            return False
        if student["Instrument"] not in room["Equipment"]:
            return False

        for lesson in self.schedule:
            if (lesson[0] == teacher_id or lesson[1] == student_id) and lesson[3] == time_slot:
                return False

        return True
