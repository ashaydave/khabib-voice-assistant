"""Utility script to create training json file for wakeword.

    There should be two directories. one that has all of the 0 labels
    and one with all the 1 labels
    
    This version recursively scans through all subdirectories.
"""
import os
import argparse
import json
import random


def collect_files_recursively(directory):
    """Recursively collects all files from a directory and its subdirectories."""
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            # Create full path to the file
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files


def main(args):
    # Recursively collect all files from the zero and one label directories
    zeros = collect_files_recursively(args.zero_label_dir)
    ones = collect_files_recursively(args.one_label_dir)
    percent = args.percent
    data = []
    
    # Add zero-labeled files
    for z in zeros:
        data.append({
            "key": z,
            "label": 0
        })
    
    # Add one-labeled files
    for o in ones:
        data.append({
            "key": o,
            "label": 1
        })
    
    # Shuffle the data
    random.shuffle(data)

    # Calculate split indices
    total = len(data)
    test_size = int(total / percent)
    train_size = total - test_size
    
    # Write train data
    train_path = os.path.join(args.save_json_path, "train.json")
    with open(train_path, 'w') as f:
        for i in range(train_size):
            line = json.dumps(data[i])
            f.write(line + "\n")
    
    # Write test data
    test_path = os.path.join(args.save_json_path, "test.json")
    with open(test_path, 'w') as f:
        for i in range(train_size, total):
            line = json.dumps(data[i])
            f.write(line + "\n")
    
    print(f"Created dataset with {total} files: {train_size} for training and {test_size} for testing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Utility script to create training json file for wakeword.

    There should be two directories. one that has all of the 0 labels
    and one with all the 1 labels
    """
    )
    parser.add_argument('--zero_label_dir', type=str, default=None, required=True,
                        help='directory of clips with zero labels')
    parser.add_argument('--one_label_dir', type=str, default=None, required=True,
                        help='directory of clips with one labels')
    parser.add_argument('--save_json_path', type=str, default=None, required=True,
                        help='path to save json file')
    parser.add_argument('--percent', type=int, default=10, required=False,
                        help='percent of clips put into test.json instead of train.json')
    args = parser.parse_args()

    main(args)