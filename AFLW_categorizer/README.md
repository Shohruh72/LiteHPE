# AFLW2000 Pose Dataset Categorizer

### This tool categorizes the AFLW2000 based on pitch, roll, or yaw pose angles. 

## 📁 Project Structure
```angular2html
project/
├── data/
│   ├── RGB/
│   │   ├── rawdata/          # Input images
│   │   └── labeldata/        # Corresponding .mat label files
├── categorized_by_pitch/     # Output directory (auto-created)
│   └── pitch_.../            # Pose range folders
│       ├── images/
│       └── labels/
├── categorize_aflw.py        # Main script
└── README.md
```
## 🔧 Features
* Supports categorization by pitch, yaw, or roll
* Automatically creates folders for each angle range
* Verifies image-label consistency
* Saves statistics as a JSON file
* Supports both CLI and interactive mode

## 🚀 Usage
### 🔹 Option 1: Command-Line

```bash 
python categorize_aflw.py \
  --image_dir data/RGB/rawdata \
  --label_dir data/RGB/labeldata \
  --output_dir data/categorized \
  --pose_type pitch
```

### Arguments
| Argument       | Description                                        | Default              |
| -------------- | -------------------------------------------------- | -------------------- |
| `--image_dir`  | Path to folder containing `.jpg` images            | `data/RGB/rawdata`   |
| `--label_dir`  | Path to folder containing `.mat` files             | `data/RGB/labeldata` |
| `--output_dir` | Destination for categorized data                   | `data/categorized`   |
| `--pose_type`  | Pose type to categorize by: `pitch`, `roll`, `yaw` | `pitch`              |

### Simply run:
```bash 
python categorize_aflw.py
```
## 📊 Output
### After running, you'll get:
* Categorized images and labels in structured subfolders
* Statistics file: categorization_statistics_<pose_type>.json
* Console summary and verification report

## ✅ Sample Output
```angular2html
Starting categorization by PITCH
...
pitch_-5_5         :    432 files (36.0%)
pitch_5_15         :    239 files (19.9%)
...
Total categorized  :   1200 files
Rejected           :     34 files
Total processed    :   1234 files

Created folders:
pitch_-5_5:
  Images: data/categorized_by_pitch/pitch_-5_5/images
  Labels: data/categorized_by_pitch/pitch_-5_5/labels
...
```

## 📁 Statistics JSON
### Example saved JSON structure:
```angular2html
{
  "pose_type": "pitch",
  "pose_ranges": {
    "pitch_-5_5": 432,
    "pitch_5_15": 239,
    ...
  },
  "rejected_count": 34,
  "total_processed": 1234,
  "total_categorized": 1200
}
```
## 🛠️ Notes
* Label files must contain a key 'Pose_Para' with pitch, yaw, and roll in radians.
* Script converts radians to degrees for comparison.
* Only .mat and .jpg file pairs are processed.
