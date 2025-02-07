import gym
import numpy as np
from gym import spaces

class SchedulingEnv(gym.Env):
    def __init__(self, teachers_df, students_df, rooms_df):
        super(SchedulingEnv, self).__init__()

        self.teachers = teachers_df
        self.students = students_df
        self.rooms = rooms_df

        # Mappings
        self.teacher_mapping = {teacher_id: idx for idx, teacher_id in enumerate(self.teachers["Teacher_ID"].values)}
        self.student_mapping = {student_id: idx for idx, student_id in enumerate(self.students["Student_ID"].values)}
        self.room_mapping = {room_id: idx for idx, room_id in enumerate(self.rooms["Room_ID"].values)}

        # Define action and observation spaces
        time_slots = 48
        self.action_space = spaces.MultiDiscrete([
            len(self.teachers),
            len(self.students),
            len(self.rooms),
            time_slots
        ])
        self.observation_space = spaces.Dict({
            "teacher_id": spaces.Discrete(len(self.teachers)),
            "student_id": spaces.Discrete(len(self.students)),
            "room_id": spaces.Discrete(len(self.rooms)),
            "time_slot": spaces.Discrete(48),
            "max_hours": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "instrument_match": spaces.Discrete(2),
            "room_equipment_match": spaces.Discrete(2),
            "student_skill": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })
        self.schedule = []

    def reset(self, seed=None, options=None):
        self.schedule = []
        return self.get_obs(), {}  # Reset must return (obs, info)

    def get_obs(self):
        teacher = self.teachers.sample(1).iloc[0]
        student = self.students.sample(1).iloc[0]
        room = self.rooms.sample(1).iloc[0]

        student_skill = (1 if student["Skill_Level"] == "Beginner"
                         else 2 if student["Skill_Level"] == "Intermediate"
                         else 3) / 3

        obs = {
            "teacher_id": self.teacher_mapping[teacher["Teacher_ID"]],
            "student_id": self.student_mapping[student["Student_ID"]],
            "room_id": self.room_mapping[room["Room_ID"]],
            "time_slot": np.random.randint(0, 48),
            "max_hours": np.array([teacher["Max_Hours_Per_Week"] / 15], dtype=np.float32),
            "instrument_match": int(student["Instrument"] in teacher["Instruments"]),
            "room_equipment_match": int(student["Instrument"] in room["Equipment"]),
            "student_skill": np.array([student_skill], dtype=np.float32)
        }
        return obs

    def step(self, action):
        teacher_idx, student_idx, room_idx, time_slot = action
        teacher_id = list(self.teacher_mapping.keys())[teacher_idx]
        student_id = list(self.student_mapping.keys())[student_idx]
        room_id = list(self.room_mapping.keys())[room_idx]

        # Retrieve teacher, student, and room information using IDs
        teacher = self.teachers[self.teachers["Teacher_ID"] == teacher_id].iloc[0]
        student = self.students[self.students["Student_ID"] == student_id].iloc[0]
        room = self.rooms[self.rooms["Room_ID"] == room_id].iloc[0]

        valid = self._is_valid_action(teacher_id, student_id, room_id, time_slot)

        reward = 0
        if valid:
            self.schedule.append((teacher_id, student_id, room_id, time_slot))
            reward += 10  # Base reward
            if student["Lessons_Per_Week"] > 2:  # Encourage scheduling high-frequency students
                reward += 5
            if student["Instrument"] in teacher["Instruments"] and student["Instrument"] in room["Equipment"]:
                reward += 10  # Extra reward for matching conditions
        else:
            reward -= 50  # Stronger penalty for invalid actions

        # Normalise reward (Min-Max Scaling)
        min_reward, max_reward = -50, 25
        normalised_reward = (reward - min_reward) / (max_reward - min_reward)

        # End episode when all students are scheduled
        done = len(self.schedule) >= len(self.students)

        return self.get_obs(), normalised_reward, done, False, {}

    def _is_valid_action(self, teacher_id, student_id, room_id, time_slot):
        teacher = self.teachers[self.teachers["Teacher_ID"] == teacher_id].iloc[0]
        student = self.students[self.students["Student_ID"] == student_id].iloc[0]
        room = self.rooms[self.rooms["Room_ID"] == room_id].iloc[0]

        if student["Instrument"] not in teacher["Instruments"] or student["Instrument"] not in room["Equipment"]:
            return False

        for lesson in self.schedule:
            if (lesson[0] == teacher_id or lesson[1] == student_id) and lesson[3] == time_slot:
                return False

        return True