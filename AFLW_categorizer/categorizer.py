import os
import numpy as np
import scipy.io as sio
from pathlib import Path
from collections import defaultdict
import json
import shutil
import argparse


class AFLW2000PoseCategorizer:
    def __init__(self, image_dir, label_dir, output_dir, pose_type='pitch'):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_dir = output_dir
        self.pose_type = pose_type.lower()

        # Define ranges for different pose types
        self.pose_ranges = self._get_pose_ranges()
        self.pose_index = self._get_pose_index()

        # Create output folders
        self.create_output_folders()

    def _get_pose_ranges(self):
        """Returns pose ranges based on pose type"""
        if self.pose_type == 'pitch':
            return {
                'pitch_-45_-35': (-45, -34),
                'pitch_-35_-25': (-35, -24),
                'pitch_-25_-15': (-25, -14),
                'pitch_-15_-5': (-15, -4),
                'pitch_-5_5': (-5, 4),
                'pitch_5_15': (5, 14),
                'pitch_15_25': (15, 24),
                'pitch_25_35': (25, 34),
                'pitch_35_45': (35, 45)
            }
        elif self.pose_type == 'roll':
            return {
                'roll_-45_-35': (-45, -34),
                'roll_-35_-25': (-35, -24),
                'roll_-25_-15': (-25, -14),
                'roll_-15_-5': (-15, -4),
                'roll_-5_5': (-5, 4),
                'roll_5_15': (5, 14),
                'roll_15_25': (15, 24),
                'roll_25_35': (25, 34),
                'roll_35_45': (35, 45)
            }
        elif self.pose_type == 'yaw':
            return {
                'yaw_-90_-75': (-90, -74),
                'yaw_-75_-60': (-75, -59),
                'yaw_-60_-45': (-60, -44),
                'yaw_-45_-35': (-45, -34),
                'yaw_-35_-25': (-35, -24),
                'yaw_-25_-15': (-25, -14),
                'yaw_-15_-5': (-15, -4),
                'yaw_-5_5': (-5, 4),
                'yaw_5_15': (5, 14),
                'yaw_15_25': (15, 24),
                'yaw_25_35': (25, 34),
                'yaw_35_45': (35, 44),
                'yaw_45_60': (45, 59),
                'yaw_60_75': (60, 74),
                'yaw_75_90': (75, 90)
            }
        else:
            raise ValueError(f"Unsupported pose type: {self.pose_type}. Use 'pitch', 'roll', or 'yaw'")

    def _get_pose_index(self):
        """Returns the index for pose parameter in Pose_Para array"""
        # Pose_Para[0] = pitch, Pose_Para[1] = yaw, Pose_Para[2] = roll
        if self.pose_type == 'pitch':
            return 0
        elif self.pose_type == 'yaw':
            return 1
        elif self.pose_type == 'roll':
            return 2

    def create_output_folders(self):
        """Creates output folders for each pose range"""
        for range_name in self.pose_ranges.keys():
            # Image folder
            image_folder = f'{self.output_dir}/{range_name}/images'
            os.makedirs(image_folder, exist_ok=True)

            # Label folder
            label_folder = f'{self.output_dir}/{range_name}/labels'
            os.makedirs(label_folder, exist_ok=True)

        print(f"Output folders created: {self.output_dir}")

    def get_pose_range(self, pose_angle):
        """Determines which range the pose angle belongs to"""
        for range_name, (min_angle, max_angle) in self.pose_ranges.items():
            if min_angle <= pose_angle <= max_angle:
                return range_name
        return None

    def categorize_dataset(self):
        """Categorizes dataset by pose ranges and copies files"""
        categorized_count = defaultdict(int)
        rejected_count = 0
        total_processed = 0

        print(f"Dataset categorization started for {self.pose_type.upper()}...")

        # Process all .mat files in label directory
        for root, dirs, files in os.walk(self.label_dir):
            for filename in files:
                if filename.endswith('.mat'):
                    total_processed += 1

                    # Mat file path
                    mat_path = os.path.join(root, filename)

                    # Corresponding image file path
                    image_filename = filename.replace('.mat', '.jpg')
                    relative_path = os.path.relpath(root, self.label_dir)

                    if relative_path == '.':
                        image_path = os.path.join(self.image_dir, image_filename)
                    else:
                        image_path = os.path.join(self.image_dir, relative_path, image_filename)

                    # Check if image file exists
                    if not os.path.exists(image_path):
                        print(f"Image not found: {image_path}")
                        rejected_count += 1
                        continue

                    try:
                        # Read pose angle from mat file
                        mat_data = sio.loadmat(mat_path)
                        pre_pose_params = mat_data['Pose_Para'][0]
                        pose_angle = pre_pose_params[self.pose_index] * 180 / np.pi  # Convert from radians to degrees

                        # Determine pose range
                        range_name = self.get_pose_range(pose_angle)

                        if range_name:
                            # Destination folders
                            dest_image_dir = f'{self.output_dir}/{range_name}/images'
                            dest_label_dir = f'{self.output_dir}/{range_name}/labels'

                            # Copy files
                            shutil.copy2(image_path, dest_image_dir)
                            shutil.copy2(mat_path, dest_label_dir)

                            categorized_count[range_name] += 1

                            if categorized_count[range_name] % 100 == 0:
                                print(f"{range_name}: {categorized_count[range_name]} files")
                        else:
                            rejected_count += 1

                    except Exception as e:
                        print(f"Error processing {mat_path}: {e}")
                        rejected_count += 1
                        continue

        # Display results
        self.print_results(categorized_count, rejected_count, total_processed)

        # Save statistics
        self.save_statistics(categorized_count, rejected_count, total_processed)

        return categorized_count

    def print_results(self, categorized_count, rejected_count, total_processed):
        """Prints categorization results"""
        print("\n" + "=" * 60)
        print(f"{self.pose_type.upper()} RANGE CATEGORIZATION RESULTS")
        print("=" * 60)

        total_categorized = sum(categorized_count.values())

        for range_name, count in categorized_count.items():
            percentage = (count / total_categorized * 100) if total_categorized > 0 else 0
            print(f"{range_name:<20}: {count:>6} files ({percentage:>5.1f}%)")

        print("-" * 60)
        print(f"{'Total categorized':<20}: {total_categorized:>6} files")
        print(f"{'Rejected':<20}: {rejected_count:>6} files")
        print(f"{'Total processed':<20}: {total_processed:>6} files")
        print("=" * 60)

        # Show created folder paths
        print("\nCreated folders:")
        for range_name in self.pose_ranges.keys():
            if categorized_count[range_name] > 0:
                print(f"{range_name}:")
                print(f"  Images: {self.output_dir}/{range_name}/images")
                print(f"  Labels: {self.output_dir}/{range_name}/labels")

    def save_statistics(self, categorized_count, rejected_count, total_processed):
        """Saves statistics to JSON file"""
        stats = {
            'pose_type': self.pose_type,
            'pose_ranges': dict(categorized_count),
            'rejected_count': rejected_count,
            'total_processed': total_processed,
            'total_categorized': sum(categorized_count.values())
        }

        stats_file = f'{self.output_dir}/categorization_statistics_{self.pose_type}.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\nStatistics saved: {stats_file}")

    def verify_categorization(self):
        """Verifies categorization results"""
        print("\n" + "=" * 50)
        print("VERIFICATION")
        print("=" * 50)

        for range_name in self.pose_ranges.keys():
            image_dir = f'{self.output_dir}/{range_name}/images'
            label_dir = f'{self.output_dir}/{range_name}/labels'

            if os.path.exists(image_dir) and os.path.exists(label_dir):
                image_count = len([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
                label_count = len([f for f in os.listdir(label_dir) if f.endswith('.mat')])

                status = "✓" if image_count == label_count else "✗"
                print(f"{range_name}: {status} Images={image_count}, Labels={label_count}")


def main():
    # Command line argument parser
    parser = argparse.ArgumentParser(description='AFLW2000 Pose Dataset Categorizer')
    parser.add_argument('--image_dir', type=str, default='data/RGB/rawdata',
                        help='Path to images directory')
    parser.add_argument('--label_dir', type=str, default='data/RGB/labeldata',
                        help='Path to labels directory')
    parser.add_argument('--output_dir', type=str, default='data/categorized',
                        help='Path to output directory')
    parser.add_argument('--pose_type', type=str, choices=['pitch', 'roll', 'yaw'], default='pitch',
                        help='Type of pose to categorize by (pitch, roll, or yaw)')

    args = parser.parse_args()

    # Update output directory to include pose type
    output_dir = f"{args.output_dir}_by_{args.pose_type}"

    print(f"Starting categorization by {args.pose_type.upper()}")
    print(f"Image directory: {args.image_dir}")
    print(f"Label directory: {args.label_dir}")
    print(f"Output directory: {output_dir}")

    # Create categorizer
    categorizer = AFLW2000PoseCategorizer(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=output_dir,
        pose_type=args.pose_type
    )

    # Categorize dataset
    results = categorizer.categorize_dataset()

    # Verify results
    categorizer.verify_categorization()

    print(f"\nCategorization completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Each {args.pose_type} range has separate 'images' and 'labels' folders")


if __name__ == "__main__":
    # If run without command line arguments, use interactive mode
    if len(os.sys.argv) == 1:
        print("AFLW2000 Pose Dataset Categorizer")
        print("=" * 40)

        # Get user input
        image_dir = input("Enter image directory path (default: data/RGB/rawdata): ").strip()
        if not image_dir:
            image_dir = "data/RGB/rawdata"

        label_dir = input("Enter label directory path (default: data/RGB/labeldata): ").strip()
        if not label_dir:
            label_dir = "data/RGB/labeldata"

        output_dir = input("Enter output directory path (default: data/categorized): ").strip()
        if not output_dir:
            output_dir = "data/categorized"

        pose_type = input("Enter pose type (pitch/roll/yaw, default: pitch): ").strip().lower()
        if pose_type not in ['pitch', 'roll', 'yaw']:
            pose_type = 'pitch'

        # Update output directory to include pose type
        output_dir = f"{output_dir}_by_{pose_type}"

        print(f"\nStarting categorization by {pose_type.upper()}")
        print(f"Image directory: {image_dir}")
        print(f"Label directory: {label_dir}")
        print(f"Output directory: {output_dir}")

        # Create categorizer
        categorizer = AFLW2000PoseCategorizer(
            image_dir=image_dir,
            label_dir=label_dir,
            output_dir=output_dir,
            pose_type=pose_type
        )

        # Categorize dataset
        results = categorizer.categorize_dataset()

        # Verify results
        categorizer.verify_categorization()

        print(f"\nCategorization completed!")
        print(f"Results saved to: {output_dir}")
        print(f"Each {pose_type} range has separate 'images' and 'labels' folders")
    else:
        main()