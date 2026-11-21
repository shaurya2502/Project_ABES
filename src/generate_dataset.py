# generate_dataset.py
# Creates the operational dataset 09_hospital_appointments.csv
# Only numpy and pandas are used. This is synthetic OPERATIONAL data (no clinical details).

import numpy as np
import pandas as pd
import os

np.random.seed(42)          # same data every time we run the script

N = 1500                    # number of unique appointments

# ---------- basic lists ----------
departments = ["General Medicine", "Pediatrics", "Orthopedics", "Cardiology",
               "Dermatology", "ENT", "Gynecology", "Ophthalmology"]
dept_weights = [0.24, 0.14, 0.12, 0.12, 0.10, 0.09, 0.10, 0.09]

base_fee = {"General Medicine": 500, "Pediatrics": 600, "Orthopedics": 900,
            "Cardiology": 1200, "Dermatology": 700, "ENT": 650,
            "Gynecology": 850, "Ophthalmology": 750}

avg_wait = {"General Medicine": 28, "Pediatrics": 22, "Orthopedics": 38,
            "Cardiology": 42, "Dermatology": 20, "ENT": 24,
            "Gynecology": 30, "Ophthalmology": 26}

noshow_base = {"General Medicine": 0.12, "Pediatrics": 0.10, "Orthopedics": 0.17,
               "Cardiology": 0.09, "Dermatology": 0.22, "ENT": 0.15,
               "Gynecology": 0.11, "Ophthalmology": 0.19}

# ---------- dates (working days only, Jan - Mar 2026) ----------
all_days = pd.date_range("2026-01-01", "2026-03-31", freq="B")   # 'B' = business days
dates = np.random.choice(all_days, size=N)

# ---------- columns ----------
dept = np.random.choice(departments, size=N, p=dept_weights)
appt_type = np.random.choice(["New", "Follow-up"], size=N, p=[0.42, 0.58])
slot = np.random.choice(["Morning", "Afternoon", "Evening"], size=N, p=[0.5, 0.35, 0.15])
sms = np.random.choice(["Yes", "No"], size=N, p=[0.6, 0.4])

df = pd.DataFrame({
    "Appointment_ID": ["APT" + str(i).zfill(5) for i in range(1, N + 1)],
    "Patient_ID": ["PAT" + str(x).zfill(4) for x in np.random.randint(1, 1100, size=N)],
    "Appointment_Date": pd.to_datetime(dates),
    "Department": dept,
    "Appointment_Type": appt_type,
    "Time_Slot": slot,
    "SMS_Reminder": sms,
})

# ---------- attendance status ----------
status = []
for i in range(N):
    p_noshow = noshow_base[df.loc[i, "Department"]]
    if df.loc[i, "SMS_Reminder"] == "Yes":
        p_noshow = p_noshow * 0.55          # reminders reduce no-shows
    else:
        p_noshow = p_noshow * 1.45
    if df.loc[i, "Appointment_Type"] == "New":
        p_noshow = p_noshow * 1.10
    p_cancel = 0.06
    p_attend = 1 - p_noshow - p_cancel
    status.append(np.random.choice(["Attended", "No-Show", "Cancelled"],
                                   p=[p_attend, p_noshow, p_cancel]))
df["Attendance_Status"] = status

# ---------- waiting time (minutes) - only exists for patients who came ----------
wait = []
for i in range(N):
    if df.loc[i, "Attendance_Status"] != "Attended":
        wait.append(np.nan)                 # nobody waited if patient did not come
    else:
        mean = avg_wait[df.loc[i, "Department"]]
        if df.loc[i, "Appointment_Type"] == "Follow-up":
            mean = mean * 0.75              # follow-ups are quicker
        if df.loc[i, "Time_Slot"] == "Afternoon":
            mean = mean * 1.15
        w = np.random.normal(mean, mean * 0.35)
        wait.append(max(2, round(w)))
wait = np.array(wait, dtype=float)

# add a few very long waits (outliers)
attended_idx = np.where(~np.isnan(wait))[0]
outlier_idx = np.random.choice(attended_idx, size=25, replace=False)
wait[outlier_idx] = np.random.randint(95, 160, size=25)

# make about 4% of ATTENDED waiting times missing (data entry gaps)
missing_idx = np.random.choice(attended_idx, size=int(0.04 * len(attended_idx)), replace=False)
wait[missing_idx] = np.nan
df["Waiting_Time_Min"] = wait

# ---------- consultation fee ----------
fees = []
for i in range(N):
    f = base_fee[df.loc[i, "Department"]]
    if df.loc[i, "Appointment_Type"] == "Follow-up":
        f = f * 0.6
    f = f + np.random.choice([-50, 0, 0, 50, 100])
    fees.append(int(f))
df["Consultation_Fee"] = fees

# ---------- add duplicate Appointment_ID rows (data quality issue) ----------
dup_rows = df.sample(12, random_state=7)
df = pd.concat([df, dup_rows], ignore_index=True)
df = df.sample(frac=1, random_state=1).reset_index(drop=True)   # shuffle rows

# ---------- save ----------
os.makedirs("data", exist_ok=True)
df["Appointment_Date"] = df["Appointment_Date"].dt.strftime("%Y-%m-%d")
df.to_csv("data/09_hospital_appointments.csv", index=False)

print("Dataset created with", len(df), "rows")
print(df.head())
