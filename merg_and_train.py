import os
import yaml
from pathlib import Path
import shutil

# 1. Define paths
project_dir = Path("/home/sachin/HazardsAndStemppedPrevention")
datasets = {
    "Industrial": project_dir / "Construction-Site-Safety-1",
    "Medical": project_dir / "Face-Mask-1",
    "Crowd": project_dir / "crowd-counting-1"
}

master_classes = []
class_map = {} # Maps (dataset_name, original_id) -> new_global_id

# 2. Build global class list and ID map
for name, path in datasets.items():
    yaml_path = path / "data.yaml"
    if not yaml_path.exists():
        print(f"Warning: {yaml_path} not found. Skipping.")
        continue
        
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        orig_classes = data['names']

    for i, class_name in enumerate(orig_classes):
        if class_name not in master_classes:
            master_classes.append(class_name)
        
        class_map[(name, i)] = master_classes.index(class_name)

print(f"Unified {len(master_classes)} classes: {master_classes}")

# 3. Create the master_data.yaml file
master_data_yaml = {
    "path": str(project_dir),
    "train": [str(d.relative_to(project_dir) / "train/images") for d in datasets.values()],
    "val": [str(d.relative_to(project_dir) / "valid/images") for d in datasets.values()],
    "nc": len(master_classes),
    "names": master_classes
}

with open(project_dir / "master_data.yaml", 'w') as f:
    yaml.dump(master_data_yaml, f)

# 4. CRITICAL: Function to remap label IDs in .txt files
def remap_labels(dataset_name, folder_path):
    label_dir = folder_path / "labels"
    if not label_dir.exists(): return
    
    for label_file in label_dir.glob("*.txt"):
        new_lines = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.split()
                if not parts: continue
                old_id = int(parts[0])
                new_id = class_map.get((dataset_name, old_id))
                if new_id is not None:
                    parts[0] = str(new_id)
                    new_lines.append(" ".join(parts))
        
        with open(label_file, 'w') as f:
            f.write("\n".join(new_lines))

print("Remapping labels to unified IDs...")
for name, path in datasets.items():
    remap_labels(name, path / "train")
    remap_labels(name, path / "valid")
    remap_labels(name, path / "test")

# 5. Start training
from ultralytics import YOLO

# Using medium model for better small object detection on RTX 5060
model = YOLO("yolov8m.pt")

results = model.train(
    data=str(project_dir / "master_data.yaml"),
    epochs=100,
    imgsz=640,
    device=0,      # Utilizes  GPU
    batch=16,      
    name="Unified_Safety_Model",
    exist_ok=True
)
print("Training completed successfully!")
