#!/usr/bin/env python3
"""
Download and prepare GAIA benchmark dataset.
This script downloads the 2023 validation set (level 1 tasks only) from Hugging Face.
"""

import json
import os
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Installing required packages...")
    os.system(f"{sys.executable} -m pip install datasets huggingface_hub")
    from datasets import load_dataset


def download_gaia_dataset():
    """Download and prepare GAIA 2023 validation level 1 tasks."""
    
    print("Downloading GAIA benchmark dataset...")
    print("Loading 2023 validation set from Hugging Face...")
    
    # Load the GAIA dataset from Hugging Face - specifically level 1 tasks
    dataset = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")
    
    # All tasks are already level 1, no filtering needed
    level_1_tasks = list(dataset)
    
    print(f"Found {len(level_1_tasks)} level 1 tasks in validation set")
    
    # Create data directory
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Process and save tasks
    processed_tasks = []
    for idx, task in enumerate(level_1_tasks):
        processed_task = {
            "task_id": task.get("task_id", f"gaia_val_l1_{idx:04d}"),
            "question": task.get("Question"),
            "level": task.get("Level"),
            "final_answer": task.get("Final answer"),
            "annotator_metadata": task.get("Annotator Metadata", {}),
            "file_name": task.get("file_name"),
            "file_path": task.get("file_path"),
        }
        processed_tasks.append(processed_task)
    
    # Save processed tasks to JSON
    output_file = data_dir / "gaia_2023_validation_level1.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_tasks, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(processed_tasks)} tasks to {output_file}")
    
    # Download any associated files
    files_dir = data_dir / "files"
    files_dir.mkdir(exist_ok=True)
    
    files_downloaded = 0
    for task in processed_tasks:
        if task.get("file_name") and task.get("file_path"):
            # Note: In actual implementation, we'd download the file from the dataset
            # For now, we just note which files are needed
            file_info_path = files_dir / f"{task['task_id']}_file_info.json"
            with open(file_info_path, "w") as f:
                json.dump({
                    "task_id": task["task_id"],
                    "file_name": task["file_name"],
                    "file_path": task["file_path"]
                }, f, indent=2)
            files_downloaded += 1
    
    if files_downloaded > 0:
        print(f"Noted {files_downloaded} tasks with associated files")
    
    # Create a summary
    summary = {
        "dataset": "GAIA 2023 Validation",
        "level": 1,
        "total_tasks": len(processed_tasks),
        "tasks_with_files": files_downloaded,
        "data_file": str(output_file.relative_to(Path(__file__).parent.parent.parent))
    }
    
    summary_file = data_dir / "dataset_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ GAIA dataset download complete!")
    print(f"Summary saved to: {summary_file}")
    
    return processed_tasks


if __name__ == "__main__":
    download_gaia_dataset()