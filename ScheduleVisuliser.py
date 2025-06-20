import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta

# Load data
data = [
    ("T004", "S001", "R04", "2025-01-01 20:30:00"),
    ("T004", "S002", "R06", "2025-01-01 16:00:00"),
    ("T005", "S003", "R07", "2025-01-01 19:00:00"),
    ("T002", "S004", "R14", "2025-01-01 08:30:00"),
    ("T005", "S005", "R05", "2025-01-01 14:30:00"),
    ("T004", "S006", "R08", "2025-01-01 10:00:00"),
    ("T005", "S007", "R04", "2025-01-01 22:00:00"),
    ("T004", "S008", "R12", "2025-01-01 09:30:00"),
    ("T004", "S009", "R15", "2025-01-01 11:30:00"),
    ("T005", "S010", "R13", "2025-01-02 05:30:00"),
    ("T004", "S011", "R01", "2025-01-01 19:30:00"),
    ("T006", "S012", "R09", "2025-01-01 14:00:00"),
    ("T005", "S013", "R15", "2025-01-02 00:30:00"),
    ("T004", "S014", "R09", "2025-01-01 23:00:00"),
    ("T005", "S015", "R10", "2025-01-02 00:30:00"),
    ("T004", "S016", "R05", "2025-01-01 18:30:00"),
    ("T004", "S017", "R08", "2025-01-02 03:30:00"),
    ("T004", "S018", "R06", "2025-01-01 22:00:00"),
    ("T004", "S019", "R15", "2025-01-01 22:30:00"),
    ("T005", "S020", "R11", "2025-01-01 15:00:00")
]

df = pd.DataFrame(data, columns=["Teacher", "Student", "Room", "Datetime"])
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Endtime'] = df['Datetime'] + timedelta(minutes=30)

# Plot
fig, ax = plt.subplots(figsize=(16, 8))
teachers = sorted(df['Teacher'].unique())
colors = plt.cm.get_cmap('tab10', len(teachers))

for i, teacher in enumerate(teachers):
    teacher_data = df[df['Teacher'] == teacher]
    for _, row in teacher_data.iterrows():
        start = mdates.date2num(row['Datetime'])
        duration = mdates.date2num(row['Endtime']) - start
        ax.broken_barh(
            [(start, duration)],
            (i, 0.8),
            facecolors=colors(i),
            edgecolor='black',
            linewidth=1.5
        )
        # Text annotation
        label = f"{row['Student']}\n{row['Room']}"
        ax.text(start + duration / 2, i + 0.4, label,
                va='center', ha='center', fontsize=8, color='white', weight='bold')

# Format axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.xticks(rotation=45)
plt.yticks([i + 0.4 for i in range(len(teachers))], teachers)
plt.title("Teacher Scheduling Calendar")
plt.xlabel("Time")
plt.ylabel("Teacher")

# Clean style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()