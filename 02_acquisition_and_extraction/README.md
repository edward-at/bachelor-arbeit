# Stage 2: acquisition and extraction

Thesis Section 4.2. This stage has no code in the repository.

The fourteen datasets were downloaded from their providers (links and access routes in the
main [README](../README.md#the-fourteen-datasets)). For each dataset, the ego-motion records were
read from the native format and written as one parquet file per clip to
`DATA_ROOT/parquet_data/<dataset>/`. A clip is one contiguous recording file; for Waymo, one
scenario.

| Dataset key | Native ego-motion source | Raw clips | Native rows |
|---|---|---:|---:|
| `4seasons` | `imu.txt` and `GNSSPoses.txt` | 30 | 31,243,319 |
| `a2d2` | bus-signal JSON | 3 | 438,131 |
| `boreas_rt` | `applanix/gps_post_process.csv` | 110 | 23,262,888 |
| `durlar` | IMU and GPS CSV from the tar parts | 4 | 1,074,550 |
| `ford` | ROS bag | 29 | 4,574,945 |
| `hdd` | per-signal timestamped files | 137 | 38,844,425 |
| `kitscenes` | `poses.txt` (8 fields) | 1,006 | 181,501 |
| `kitti` | OXTS text | 156 | 47,962 |
| `kitti360` | OXTS text | 10 | 940,262 |
| `nuscene` | JSON (`ms_imu`, `pose`, `ego_pose`) | 979 | 1,906,954 |
| `nvidia` | `egomotion.offline` parquet (pose and quaternion) | 298,326 | 60,246,690 |
| `oxford` | `rtk.csv` | 70 | 1,355,769 |
| `waymo` | TFRecord, one scenario per clip | 70,541 | 14,000,081 |
| `zod` | `oxts.hdf5` | 29 | 661,429 |
| **Total** | | **371,430** | **178,778,906** |

Every clip file carries a nanosecond timestamp and the position, velocity, acceleration and
attitude columns the dataset provides, at the native rate. The per-dataset loaders in
`03_standardization/standardize_inertial_data.py` document which columns are read and how units
and frames are converted (for example the HDD latitude and longitude order, the A2D2 speed unit
and the ZOD yaw-rate unit).
