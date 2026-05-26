# `data/` — Dataset & MongoDB schema documentation

This folder contains the input dataset for the project and documents the
MongoDB document schema that the ingestion pipeline produces.

## Folder layout

```
data/
├── README.md                  (this file)
└── PAMAP2_Dataset/
    ├── readme.pdf
    ├── DataCollectionProtocol.pdf
    ├── DescriptionOfActivities.pdf
    ├── PerformedActivitiesSummary.pdf
    ├── subjectInformation.pdf
    ├── Protocol/              (9 subjects, the 12 required activities)
    │   ├── subject101.dat
    │   ├── ...
    │   └── subject109.dat
    └── Optional/              (subset of subjects, additional activities)
```

Do **not** rename or restructure the `PAMAP2_Dataset/` directory.

## Raw `.dat` format

Each `.dat` file is a whitespace-separated text file with **54 columns** sampled
at **100 Hz**. The relevant column groups (1-indexed):

| Columns | Content |
|---------|---------|
| 1 | timestamp (seconds) |
| 2 | activity ID (see table in root `README.md`) |
| 3 | heart rate (bpm) — **not used** (biometric, sampled at ~9 Hz) |
| 4–20  | IMU **hand/wrist**: temp, acc±16g, acc±6g, gyro, mag, orientation |
| 21–37 | IMU **chest**: same layout |
| 38–54 | IMU **ankle**: same layout |

For each IMU we keep the **±16 g accelerometer** (3 axes) and the **gyroscope**
(3 axes) — 6 channels per IMU, sampled natively at 100 Hz. Heart rate,
magnetometer, ±6 g accelerometer, temperature and orientation are skipped per
the project's Sensor Selection Strategy ("inertial sensing only").

Missing samples (dropped wireless packets) are encoded as `NaN` and handled
by linear interpolation in the ingestion notebook
(`aiot_dataset_creation.ipynb`).

## Ingestion data-flow

```
.dat files ──parse──▶ per-(subject, activity, IMU) segments
            ──insert──▶ MongoDB collection `aiot_course.pamap2`
            ──fetch───▶ aiot_project_*.ipynb
```

Concretely:

1. `pandas` reads each `.dat` file (only the required columns via `usecols`).
2. `NaN` values are filled with linear interpolation; residual leading `NaN`
   is dropped.
3. Rows where `activity_id == 0` (transient periods between activities) are
   discarded.
4. The remaining rows are split into **contiguous (subject, activity)
   segments** (every change in `activity_id` starts a new segment).
5. For each segment, **one document per IMU location** (`hand`, `chest`,
   `ankle`) is inserted, following the schema below.

The analysis notebooks (`aiot_project_time_series.ipynb`,
`aiot_project_feature_engineering.ipynb`) **fetch documents from MongoDB**
rather than re-reading the raw files.

## MongoDB document schema

Each document corresponds to one contiguous segment of one activity performed
by one subject, recorded from one IMU location. The default configuration is
`AccGyr` (accelerometer + gyroscope):

```json
{
  "_id": ObjectId("6984b3fa87abe7f4dff571aa"),
  "data": {
    "acc_x": ["array", "of", "values"],
    "acc_y": ["array", "of", "values"],
    "acc_z": ["array", "of", "values"],
    "gyr_x": ["array", "of", "values"],
    "gyr_y": ["array", "of", "values"],
    "gyr_z": ["array", "of", "values"]
  },
  "activity_id": 4,
  "activity_label": "walking",
  "subject": "101",
  "split": "Protocol",
  "imu_location": "hand",
  "sensor": "AccGyr",
  "sr": 100,
  "datetime": "MongoDB datetime object generated with datetime.datetime.now()"
}
```

### Field notes

- `activity_id` — the integer label as it appears in the `.dat` file.
- `activity_label` — human-readable name (see the activity table in the root
  `README.md`).
- `subject` — file's subject identifier as a string (`"101"` … `"109"`).
- `split` — `"Protocol"` or `"Optional"`, matching the source subdirectory.
  The current project ingests `Protocol` only.
- `imu_location` — `"hand"`, `"chest"`, or `"ankle"`. **One document per IMU
  location** — IMUs are not concatenated into a single document.
- `sensor` — `"Acc"`, `"Gyr"`, or `"AccGyr"`. If the magnetometer is added,
  the convention is `"AccGyrMag"` with extra `mag_x`, `mag_y`, `mag_z` keys.
- `sr` — sampling rate in Hz. Always `100` in this project (no down-sampling
  was needed because every channel kept is natively at 100 Hz).

## Reproduction

To restore the collection on another machine:

```bash
mongorestore --db aiot_course mongodump_har/aiot_course
```

To regenerate the collection from the raw `.dat` files:

```bash
py -3.11 -m jupyter nbconvert --to notebook --execute --inplace aiot_dataset_creation.ipynb
```
