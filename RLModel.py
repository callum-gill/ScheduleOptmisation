import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config import MAX_TEACHERS, MAX_ROOMS, MAX_STUDENTS, TIME_SLOTS

class SchedulingEnv(gym.Env):
    def __init__(self, teachers_df, students_df, rooms_df, max_steps=1000, target_lessons=None):
        super(SchedulingEnv, self).__init__()

        self.teachers = teachers_df.reset_index(drop=True)
        self.students = students_df.reset_index(drop=True)
        self.rooms = rooms_df.reset_index(drop=True)

        self.max_steps = max_steps
        self.target_lessons = target_lessons or len(self.students)
        self.steps = 0
        self.current_student_index = 0

        # Mappings
        self.teacher_ids = self.teachers["Teacher_ID"].tolist()
        self.student_ids = self.students["Student_ID"].tolist()
        self.room_ids = self.rooms["Room_ID"].tolist()

        self.teacher_mapping = {tid: idx for idx, tid in enumerate(self.teacher_ids)}
        self.student_mapping = {sid: idx for idx, sid in enumerate(self.student_ids)}
        self.room_mapping = {rid: idx for idx, rid in enumerate(self.room_ids)}

        # New action space: teacher, room, time slot (no student)
        self.action_space = spaces.MultiDiscrete([
            MAX_TEACHERS,
            MAX_ROOMS,
            TIME_SLOTS
        ])

        # Observation: current student info (scaled)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(27,), dtype=np.float32
        )
        self.last_obs = []

        self.schedule = []

    def reset(self, **kwargs):
        self.steps = 0
        self.schedule = []
        self.current_student_index = 0
        self.last_obs = []
        return self.get_obs(), {}

    def get_obs(self):
        if self.current_student_index >= len(self.student_ids):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        student_id = self.student_ids[self.current_student_index]
        student = self.students[self.students["Student_ID"] == student_id].iloc[0]
        instrument = student["Instrument"]

        # Teacher compatibility mask (binary vector)
        teacher_mask = np.array([
            1.0 if instrument in t_instruments else 0.0
            for t_instruments in self.teachers["Instruments"]
        ], dtype=np.float32)

        # Binary vector of scheduled students (global state)
        scheduled_mask = np.zeros(len(self.student_ids), dtype=np.float32)
        for lesson in self.schedule:
            student_idx = self.student_mapping[lesson[1]]
            scheduled_mask[student_idx] = 1.0

        obs = np.concatenate([
            teacher_mask,
            scheduled_mask
        ])

        return obs

    def step(self, action):
        self.steps += 1
        teacher_idx, room_idx, time_slot = action

        # Validate indexes
        if teacher_idx >= len(self.teacher_ids) or \
           room_idx >= len(self.room_ids) or \
           self.current_student_index >= len(self.student_ids):
            return self.last_obs, -1.0, True, False, {"error": "index out of bounds"}

        teacher_id = self.teacher_ids[teacher_idx]
        student_id = self.student_ids[self.current_student_index]
        room_id = self.room_ids[room_idx]

        reward = 0
        new_lesson = None
        info = {}

        if self._is_valid_action(teacher_id, student_id, room_id, time_slot):
            lesson = (teacher_id, student_id, room_id, time_slot)
            print(action)

            if lesson not in self.schedule:
                self.current_student_index += 1

                # Track existing status before update
                prev_student_lessons = [l for l in self.schedule if l[1] == student_id]
                already_scheduled = len(prev_student_lessons) > 0
                prev_pair_exists = any(l[0] == teacher_id and l[1] == student_id for l in self.schedule)

                # Add lesson to schedule
                self.schedule.append(lesson)
                new_lesson = lesson

                # Reward components
                reward += 1.0  # ✅ Base reward

                if not already_scheduled:
                    print("📌 First time student scheduled")
                    reward += 2.0

                if not prev_pair_exists:
                    print("👥 New teacher-student pair")
                    reward += 0.5
            else:
                print("🛑 Duplicate scheduling penalty")
                reward -= 1.0
        else:
            print("❌ Invalid action")
            reward -= 10.0
            info["error"] = "invalid action"

        all_scheduled = len(set(l[1] for l in self.schedule)) == len(self.student_ids)
        done = self.steps >= self.max_steps or all_scheduled
        truncated = self.steps >= self.max_steps

        if all_scheduled:
            print("✅ All student scheduled")
            reward += 5.0

        unique_students = len(set(l[1] for l in self.schedule))
        info["coverage"] = unique_students / len(self.student_ids)
        info["new_lesson"] = new_lesson

        print(f"✅ Coverage: {len(set(l[1] for l in self.schedule))} / {len(self.student_ids)}")

        return self.get_obs(), reward, done, truncated, info

    def _is_valid_action(self, teacher_id, student_id, room_id, time_slot):
        teacher = self.teachers[self.teachers["Teacher_ID"] == teacher_id].iloc[0]
        student = self.students[self.students["Student_ID"] == student_id].iloc[0]

        if student["Instrument"] not in teacher["Instruments"]:
            return False

        if any(
            lesson[3] == time_slot and (
                lesson[0] == teacher_id or
                lesson[1] == student_id or
                lesson[2] == room_id
            )
            for lesson in self.schedule
        ):
            return False

        return True

    def decode_action(self, action):
        teacher_idx, room_idx, timeslot = action

        if teacher_idx > len(self.teacher_ids) or room_idx > len(self.room_ids):
            return []

        teacher_id = self.teacher_ids[teacher_idx]
        room_id = self.room_ids[room_idx]
        student_id = self.student_ids[self.current_student_index]
        return teacher_id, student_id, room_id, timeslot