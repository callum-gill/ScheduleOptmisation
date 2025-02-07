import gym
import numpy as np
import random
from gym import spaces


class SchedulingEnv(gym.Env):
    def __init__(self, teachers_df, students_df, rooms_df):
        super(SchedulingEnv, self).__init__()

        self.teachers = teachers_df
        self.students = students_df
        self.rooms = rooms_df

        # Create mappings for categorical variables
        self.teacher_mapping = {teacher_id: idx for idx, teacher_id in enumerate(self.teachers["Teacher_ID"].values)}
        self.student_mapping = {student_id: idx for idx, student_id in enumerate(self.students["Student_ID"].values)}
        self.room_mapping = {room_id: idx for idx, room_id in enumerate(self.rooms["Room_ID"].values)}

        # Define action and observation spaces
        time_slots = 48
        self.action_space = spaces.MultiDiscrete([
            len(self.teachers),  # Teacher selection
            len(self.students),  # Student selection
            len(self.rooms),  # Room selection
            time_slots  # Time slot selection
        ])
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)

        self.schedule = []


    def seed(self, seed=None):
        """Set the seed for reproducibility."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)  # Use Gym's seeding utility
        random.seed(seed)
        np.random.seed(seed)
        return [seed]


    def reset(self, **kwargs):
        # Reset the schedule and available resources
        self.schedule = []
        return self.get_obs()

    def get_obs(self):
        teacher = self.teachers.sample(1).iloc[0]
        student = self.students.sample(1).iloc[0]
        room = self.rooms.sample(1).iloc[0]

        # One-hot encode skill level (Beginner=1, Intermediate=2, Advanced=3) and normalize
        student_skill = (1 if student["Skill_Level"] == "Beginner"
                         else 2 if student["Skill_Level"] == "Intermediate"
        else 3) / 3

        obs = np.array([
            teacher["Max_Hours_Per_Week"] / 15,  # Normalize max hours
            len(teacher["Instruments"]) / 4,  # Normalize instrument count
            len(room["Equipment"]) / 4,  # Normalize equipment count
            1 if student["Instrument"] in teacher["Instruments"] else 0,
            1 if student["Instrument"] in room["Equipment"] else 0,
            student["Lessons_Per_Week"] / 3,  # Normalize lessons per week
            student_skill,  # Normalized one-hot encoded skill level
            np.random.randint(0, 48) / 48  # Normalize time slot
        ], dtype=np.float32)

        return obs


    def step(self, action):
        # Decode the action
        teacher_idx, student_idx, room_idx, time_slot = action  # Action is now a list of 4 values

        # Retrieve IDs using indices
        teacher_id = list(self.teacher_mapping.keys())[teacher_idx]
        student_id = list(self.student_mapping.keys())[student_idx]
        room_id = list(self.room_mapping.keys())[room_idx]

        # Validate the action
        valid = self._is_valid_action(teacher_id, student_id, room_id, time_slot)
        reward = 10 if valid else -10

        if valid:
            self.schedule.append((teacher_id, student_id, room_id, time_slot))

        # Check if done (e.g., all lessons assigned or max steps reached)
        done = len(self.schedule) >= len(self.students)
        info = {
            "teacher": teacher_id,
            "student": student_id,
            "room": room_id,
            "time_slot": time_slot,
            "valid": valid
        }
        return self.get_obs(), reward, done, info


    def _is_valid_action(self, teacher_id, student_id, room_id, time_slot):
        teacher = self.teachers[self.teachers["Teacher_ID"] == teacher_id].iloc[0]
        student = self.students[self.students["Student_ID"] == student_id].iloc[0]
        room = self.rooms[self.rooms["Room_ID"] == room_id].iloc[0]

        # Instrument compatibility
        if student["Instrument"] not in teacher["Instruments"] or student["Instrument"] not in room["Equipment"]:
            return False

        # Check if teacher or student is already scheduled at this time
        for lesson in self.schedule:
            if (lesson[0] == teacher_id or lesson[1] == student_id) and lesson[3] == time_slot:
                return False  # Teacher or student already booked

        return True