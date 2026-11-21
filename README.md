# Hospital Appointment Operations Analysis

Analysis of appointment attendance, waiting times and department workload to find operational bottlenecks.
Built with **Python, NumPy and Pandas**. This is operational data only, with no medical conclusions.

## Project Files
- `generate_dataset.py` creates the dataset (`data/09_hospital_appointments.csv`)
- `analysis.py` runs all 15 analysis tasks
- `outputs/` holds all results (CSV tables and `analysis_report.txt`)
- `docs/SRS.docx` is the requirements document

## What It Does
- Cleans data: missing waiting times and duplicate Appointment_IDs
- Attendance and no-show analysis by department and SMS reminder
- Waiting times by department and appointment type, with IQR outliers and waiting-time bands
- Consultation fee summary and estimated fees for attended appointments
- Five operational recommendations

## How to Run
```
pip install -r requirements.txt
python generate_dataset.py
python analysis.py
```
