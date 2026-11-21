# analysis.py
# Project 9: Hospital Appointment Operations Analysis
# Libraries used: numpy and pandas only.
# NOTE: This is an operational analysis. No medical conclusions are made.

import numpy as np
import pandas as pd
import os

# all results are saved inside ONE folder called "outputs"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

report = []   # every printed section is also stored here and written to a text file


def show(title, obj):
    """Print a section and keep a copy for the report file."""
    text = obj.to_string() if hasattr(obj, "to_string") else str(obj)
    block = "\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70 + "\n" + text
    print(block)
    report.append(block)


# ----------------------------------------------------------------------
# Task 1: Load and inspect appointment records
# ----------------------------------------------------------------------
df = pd.read_csv("data/09_hospital_appointments.csv")
df["Appointment_Date"] = pd.to_datetime(df["Appointment_Date"])

show("TASK 1a: First 5 rows", df.head())
show("TASK 1b: Shape (rows, columns)", df.shape)
show("TASK 1c: Data types", df.dtypes)
show("TASK 1d: Summary statistics (numeric columns)", df.describe())

# ----------------------------------------------------------------------
# Task 2: Missing waiting times
# ----------------------------------------------------------------------
missing_per_col = df.isnull().sum()
show("TASK 2a: Missing values per column", missing_per_col)

# Waiting time is blank for No-Show / Cancelled patients because nobody waited.
# That is "not applicable". Only blanks for ATTENDED patients are real missing data.
not_applicable = df["Waiting_Time_Min"].isnull() & (df["Attendance_Status"] != "Attended")
real_missing = df["Waiting_Time_Min"].isnull() & (df["Attendance_Status"] == "Attended")

summary_missing = pd.Series({
    "Total blank waiting times": df["Waiting_Time_Min"].isnull().sum(),
    "Not applicable (No-Show / Cancelled)": not_applicable.sum(),
    "Real missing (Attended patients)": real_missing.sum(),
})
show("TASK 2b: Type of missing waiting times", summary_missing)

# Handling: fill real missing values with the MEDIAN of the same department + appointment type.
# (median is used because waiting time has outliers)
df["Waiting_Time_Clean"] = df["Waiting_Time_Min"]
group_median = df.groupby(["Department", "Appointment_Type"])["Waiting_Time_Min"].transform("median")
df.loc[real_missing, "Waiting_Time_Clean"] = group_median[real_missing]
show("TASK 2c: Real missing values left after filling",
     df.loc[df["Attendance_Status"] == "Attended", "Waiting_Time_Clean"].isnull().sum())

# ----------------------------------------------------------------------
# Task 3: Duplicate Appointment_ID values
# ----------------------------------------------------------------------
dup_ids = df["Appointment_ID"].duplicated().sum()
dup_full_rows = df.duplicated().sum()
show("TASK 3a: Duplicate Appointment_ID count", dup_ids)
show("TASK 3b: Completely identical duplicate rows", dup_full_rows)

duplicates_view = df[df["Appointment_ID"].duplicated(keep=False)].sort_values("Appointment_ID")
duplicates_view.to_csv(f"{OUT}/duplicate_appointments_found.csv", index=False)

df = df.drop_duplicates(subset="Appointment_ID", keep="first").reset_index(drop=True)
show("TASK 3c: Rows after removing duplicates", df.shape[0])

# ----------------------------------------------------------------------
# Task 4: Total appointments and attendance-status percentages
# ----------------------------------------------------------------------
total_appts = len(df)
status_counts = df["Attendance_Status"].value_counts()
status_pct = (df["Attendance_Status"].value_counts(normalize=True) * 100).round(2)
status_table = pd.DataFrame({"Count": status_counts, "Percentage": status_pct})

show("TASK 4a: Total appointments", total_appts)
show("TASK 4b: Attendance status counts and percentages", status_table)
status_table.to_csv(f"{OUT}/attendance_status_summary.csv")

# ----------------------------------------------------------------------
# Task 5: Average waiting time by department
# ----------------------------------------------------------------------
attended = df[df["Attendance_Status"] == "Attended"].copy()

wait_by_dept = attended.groupby("Department")["Waiting_Time_Clean"].agg(
    Attended_Appointments="count",
    Avg_Wait_Min="mean",
    Max_Wait_Min="max"
).round(2).sort_values("Avg_Wait_Min", ascending=False)

show("TASK 5: Average waiting time by department (attended patients)", wait_by_dept)
wait_by_dept.to_csv(f"{OUT}/avg_waiting_time_by_department.csv")

# ----------------------------------------------------------------------
# Task 6: Departments with the highest no-show percentage
# ----------------------------------------------------------------------
df["Is_NoShow"] = (df["Attendance_Status"] == "No-Show").astype(int)

noshow_by_dept = df.groupby("Department").agg(
    Total_Appointments=("Appointment_ID", "count"),
    NoShow_Count=("Is_NoShow", "sum"),
    NoShow_Percent=("Is_NoShow", "mean")
)
noshow_by_dept["NoShow_Percent"] = (noshow_by_dept["NoShow_Percent"] * 100).round(2)
noshow_by_dept = noshow_by_dept.sort_values("NoShow_Percent", ascending=False)

show("TASK 6: No-show percentage by department (highest first)", noshow_by_dept)
noshow_by_dept.to_csv(f"{OUT}/noshow_by_department.csv")

# ----------------------------------------------------------------------
# Task 7: SMS reminder Yes vs No (crosstab)
# ----------------------------------------------------------------------
sms_counts = pd.crosstab(df["SMS_Reminder"], df["Attendance_Status"])
sms_pct = (pd.crosstab(df["SMS_Reminder"], df["Attendance_Status"], normalize="index") * 100).round(2)

show("TASK 7a: SMS reminder vs attendance status (counts)", sms_counts)
show("TASK 7b: SMS reminder vs attendance status (row %)", sms_pct)
sms_pct.to_csv(f"{OUT}/sms_reminder_vs_attendance_percent.csv")

sms_noshow = sms_pct["No-Show"]
sms_difference = round(sms_noshow["No"] - sms_noshow["Yes"], 2)
show("TASK 7c: No-show % difference (No SMS minus SMS)", sms_difference)

# ----------------------------------------------------------------------
# Task 8: Average consultation fee by department
# ----------------------------------------------------------------------
fee_by_dept = df.groupby("Department")["Consultation_Fee"].agg(
    Avg_Fee="mean", Min_Fee="min", Max_Fee="max"
).round(2).sort_values("Avg_Fee", ascending=False)

show("TASK 8: Average consultation fee by department", fee_by_dept)
fee_by_dept.to_csv(f"{OUT}/avg_fee_by_department.csv")

# ----------------------------------------------------------------------
# Task 9: Appointment volume by date and department
# ----------------------------------------------------------------------
volume_date_dept = pd.crosstab(df["Appointment_Date"], df["Department"])
volume_date_dept["Total"] = volume_date_dept.sum(axis=1)
volume_date_dept.to_csv(f"{OUT}/volume_by_date_and_department.csv")
show("TASK 9a: Volume by date and department (first 10 days)", volume_date_dept.head(10))

daily_total = df.groupby("Appointment_Date").size()
show("TASK 9b: Busiest 5 dates", daily_total.sort_values(ascending=False).head(5))

df["Weekday"] = df["Appointment_Date"].dt.day_name()
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
weekday_volume = pd.crosstab(df["Weekday"], df["Department"]).reindex(weekday_order)
show("TASK 9c: Volume by weekday and department", weekday_volume)
weekday_volume.to_csv(f"{OUT}/volume_by_weekday_and_department.csv")

df["Month"] = df["Appointment_Date"].dt.strftime("%Y-%m")
monthly_volume = pd.crosstab(df["Month"], df["Department"])
show("TASK 9d: Monthly volume by department", monthly_volume)

dept_workload = df["Department"].value_counts()
show("TASK 9e: Total workload per department", dept_workload)

# ----------------------------------------------------------------------
# Task 10: New vs follow-up waiting times
# ----------------------------------------------------------------------
wait_by_type = attended.groupby("Appointment_Type")["Waiting_Time_Clean"].agg(
    Count="count", Avg_Wait="mean", Median_Wait="median", Max_Wait="max"
).round(2)
show("TASK 10a: Waiting time - New vs Follow-up", wait_by_type)
wait_by_type.to_csv(f"{OUT}/waiting_time_new_vs_followup.csv")

wait_type_dept = attended.pivot_table(index="Department", columns="Appointment_Type",
                                      values="Waiting_Time_Clean", aggfunc="mean").round(2)
show("TASK 10b: Average waiting time by department and appointment type", wait_type_dept)

# ----------------------------------------------------------------------
# Task 11: NumPy - median, quartiles, standard deviation
# ----------------------------------------------------------------------
wait_array = attended["Waiting_Time_Clean"].to_numpy()

median_wait = np.median(wait_array)
q1 = np.percentile(wait_array, 25)
q2 = np.percentile(wait_array, 50)
q3 = np.percentile(wait_array, 75)
std_wait = np.std(wait_array)
mean_wait = np.mean(wait_array)

numpy_stats = pd.Series({
    "Mean": round(mean_wait, 2),
    "Median": median_wait,
    "Q1 (25%)": q1,
    "Q2 (50%)": q2,
    "Q3 (75%)": q3,
    "Standard deviation": round(std_wait, 2),
    "Minimum": np.min(wait_array),
    "Maximum": np.max(wait_array),
})
show("TASK 11: NumPy statistics of waiting time (minutes)", numpy_stats)
numpy_stats.to_csv(f"{OUT}/waiting_time_numpy_stats.csv", header=["Value"])

# ----------------------------------------------------------------------
# Task 12: Unusually long waiting times using IQR
# ----------------------------------------------------------------------
iqr = q3 - q1
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr

outliers = attended[(attended["Waiting_Time_Clean"] > upper_limit) |
                    (attended["Waiting_Time_Clean"] < lower_limit)]

iqr_info = pd.Series({
    "IQR": iqr,
    "Lower limit": lower_limit,
    "Upper limit": upper_limit,
    "Number of outliers": len(outliers),
    "Outlier percent of attended": round(len(outliers) / len(attended) * 100, 2),
})
show("TASK 12a: IQR outlier limits", iqr_info)

outliers_by_dept = outliers["Department"].value_counts()
show("TASK 12b: Long-wait outliers by department", outliers_by_dept)
outliers.sort_values("Waiting_Time_Clean", ascending=False).to_csv(
    f"{OUT}/long_wait_outliers.csv", index=False)

# ----------------------------------------------------------------------
# Task 13: Waiting-time categories
# ----------------------------------------------------------------------
bins = [0, 15, 30, 45, 60, np.inf]
labels = ["0-15 min (Short)", "16-30 min (Moderate)", "31-45 min (Long)",
          "46-60 min (Very Long)", "60+ min (Critical)"]
attended["Wait_Band"] = pd.cut(attended["Waiting_Time_Clean"], bins=bins, labels=labels)

band_counts = attended["Wait_Band"].value_counts().sort_index()
band_table = pd.DataFrame({
    "Count": band_counts,
    "Percentage": (band_counts / band_counts.sum() * 100).round(2)
})
show("TASK 13a: Waiting-time bands", band_table)
band_table.to_csv(f"{OUT}/waiting_time_bands.csv")

band_by_dept = pd.crosstab(attended["Department"], attended["Wait_Band"])
show("TASK 13b: Waiting-time bands by department", band_by_dept)
band_by_dept.to_csv(f"{OUT}/waiting_time_bands_by_department.csv")

# ----------------------------------------------------------------------
# Task 14: Estimated total consultation fees (attended only)
# ----------------------------------------------------------------------
total_fee = attended["Consultation_Fee"].sum()
fee_summary = attended.groupby("Department")["Consultation_Fee"].agg(
    Attended_Appointments="count", Total_Fees="sum", Avg_Fee="mean"
).round(2).sort_values("Total_Fees", ascending=False)

show("TASK 14a: Estimated total fees (attended appointments)", total_fee)
show("TASK 14b: Estimated fees by department", fee_summary)
fee_summary.to_csv(f"{OUT}/estimated_fees_attended.csv")

# extra: fee value lost because of no-shows
lost_fee = df[df["Attendance_Status"] == "No-Show"]["Consultation_Fee"].sum()
show("EXTRA: Estimated fee value lost to no-shows", lost_fee)

# ----------------------------------------------------------------------
# Extra analysis: time slot workload
# ----------------------------------------------------------------------
slot_summary = df.groupby("Time_Slot").agg(
    Appointments=("Appointment_ID", "count"),
    NoShow_Percent=("Is_NoShow", lambda x: round(x.mean() * 100, 2))
)
slot_wait = attended.groupby("Time_Slot")["Waiting_Time_Clean"].mean().round(2)
slot_summary["Avg_Wait_Min"] = slot_wait
show("EXTRA: Time slot summary", slot_summary)
slot_summary.to_csv(f"{OUT}/time_slot_summary.csv")

# ----------------------------------------------------------------------
# Department summary (main deliverable)
# ----------------------------------------------------------------------
dept_summary = pd.DataFrame({
    "Total_Appointments": df.groupby("Department").size(),
    "NoShow_Percent": noshow_by_dept["NoShow_Percent"],
    "Avg_Wait_Min": wait_by_dept["Avg_Wait_Min"],
    "Avg_Fee": fee_by_dept["Avg_Fee"],
    "Est_Fees_Attended": fee_summary["Total_Fees"],
}).sort_values("Total_Appointments", ascending=False)
show("DELIVERABLE: Department summary", dept_summary)
dept_summary.to_csv(f"{OUT}/department_summary.csv")

# ----------------------------------------------------------------------
# Task 15: Five operational recommendations (built from the results)
# ----------------------------------------------------------------------
top_noshow = noshow_by_dept.index[0]
top_noshow_pct = noshow_by_dept["NoShow_Percent"].iloc[0]
second_noshow = noshow_by_dept.index[1]
longest_wait_dept = wait_by_dept.index[0]
longest_wait_val = wait_by_dept["Avg_Wait_Min"].iloc[0]
busiest_dept = dept_workload.index[0]
busiest_day = weekday_volume.sum(axis=1).idxmax()
slot_longest = slot_summary["Avg_Wait_Min"].idxmax()
long_pct = band_table["Percentage"].iloc[3] + band_table["Percentage"].iloc[4]
newfu_gap = round(wait_by_type.loc["New", "Avg_Wait"] - wait_by_type.loc["Follow-up", "Avg_Wait"], 2)

recommendations = [
    f"1. Send SMS reminders for every appointment. No-show rate is {sms_noshow['No']}% without "
    f"an SMS and {sms_noshow['Yes']}% with one, a gap of {sms_difference} percentage points.",
    f"2. Focus no-show reduction on {top_noshow} ({top_noshow_pct}%) and {second_noshow}. "
    f"Use confirmation calls and a small overbooking buffer in these departments.",
    f"3. Reduce waiting at {longest_wait_dept}, which has the longest average wait "
    f"({longest_wait_val} min). Review doctor schedules and add support staff in busy hours.",
    f"4. Balance workload: {busiest_dept} has the most appointments and {busiest_day} is the busiest "
    f"weekday. Spread bookings across other days and add staff on peak days.",
    f"5. Monitor waits above {upper_limit:.0f} minutes (IQR limit) and the {slot_longest} slot, which has "
    f"the longest average wait. {long_pct:.1f}% of attended patients wait over 45 minutes. "
    f"New patients wait {newfu_gap} min more than follow-ups, so give new patients dedicated slots.",
]
show("TASK 15: Operational recommendations", "\n".join(recommendations))

# ----------------------------------------------------------------------
# Save cleaned data and full text report
# ----------------------------------------------------------------------
df.to_csv(f"{OUT}/cleaned_appointments.csv", index=False)

with open(f"{OUT}/analysis_report.txt", "w") as f:
    f.write("HOSPITAL APPOINTMENT OPERATIONS ANALYSIS - FULL REPORT\n")
    f.write("(Operational data only. No medical conclusions.)\n")
    f.write("\n".join(report))

print("\nAll output files saved in the folder:", OUT)
