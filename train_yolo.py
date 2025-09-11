import os
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class MultiDatasetYOLOProcessor:
    def __init__(self, output_dir="merged_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create directory structure
        (self.output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # Combined class mapping
        self.class_mapping = {
            # Car accident classes (0-4)
            'no_accident': 0,
            'minor_accident': 1, 
            'moderate_accident': 2,
            'severe_accident': 3,
            'totaled_vehicle': 4,
            # Pothole class (5) - simplified to single class
            'pothole': 5
        }
        
        self.class_names = list(self.class_mapping.keys())
        
    def convert_xml_to_yolo(self, xml_file, img_width, img_height):
        """Convert XML annotation to YOLO format"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            yolo_annotations = []
            
            # Look for objects (might be named differently in different datasets)
            objects = root.findall('object')
            if not objects:
                # Try alternative names
                objects = root.findall('annotation/object') or root.findall('.//object')
            
            print(f"    Found {len(objects)} objects in {xml_file.name}")
            
            for obj in objects:
                # For potholes, use single class regardless of size
                bbox = obj.find('bndbox')
                if bbox is None:
                    print(f"    WARNING: No bndbox found in object")
                    continue
                
                try:
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                except (AttributeError, ValueError) as e:
                    print(f"    ERROR: Could not parse bbox coordinates: {e}")
                    continue
                
                # All potholes get the same class ID
                class_id = self.class_mapping['pothole']
                
                # Convert to YOLO format (normalized coordinates)
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                # Ensure coordinates are within bounds
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                print(f"    Added pothole annotation: bbox=({xmin:.1f},{ymin:.1f},{xmax:.1f},{ymax:.1f})")
            
            return yolo_annotations
            
        except Exception as e:
            print(f"    ERROR converting XML {xml_file.name}: {e}")
            return []
    
    def process_car_accident_dataset(self, dataset_path, split_ratio=0.8):
        """Process car accident dataset (already in YOLO format)"""
        dataset_path = Path(dataset_path)
        
        # Copy images and labels
        for split in ['train', 'val']:
            img_dir = dataset_path / "images" / split
            label_dir = dataset_path / "labels" / split
            
            if img_dir.exists() and label_dir.exists():
                for img_file in img_dir.glob("*.jpg"):
                    label_file = label_dir / f"{img_file.stem}.txt"
                    
                    if label_file.exists():
                        # Copy image
                        shutil.copy2(img_file, self.output_dir / "images" / split / f"car_{img_file.name}")
                        
                        # Copy label (classes already mapped correctly 0-4)
                        shutil.copy2(label_file, self.output_dir / "labels" / split / f"car_{img_file.stem}.txt")
    
    def process_pothole_dataset_yolo_format(self, dataset_path):
        """Process pothole dataset that's already in YOLO format"""
        dataset_path = Path(dataset_path)
        
        print(f"Looking for pothole dataset at: {dataset_path.absolute()}")
        
        # Check if paths exist
        if not dataset_path.exists():
            print(f"ERROR: Dataset path does not exist: {dataset_path}")
            return
        
        train_dir = dataset_path / "train"
        test_dir = dataset_path / "test"
        
        if not train_dir.exists():
            print(f"ERROR: train directory not found: {train_dir}")
            return
            
        if not test_dir.exists():
            print(f"ERROR: test directory not found: {test_dir}")
            return
        
        print(f"Found train directory with {len(list(train_dir.glob('*')))} files")
        print(f"Found test directory with {len(list(test_dir.glob('*')))} files")
        
        total_processed = 0
        total_annotations = 0
        
        # Process train and test splits
        splits = {'train': train_dir, 'test': test_dir}
        
        for split_name, split_dir in splits.items():
            # Map 'test' to 'val' for consistency with YOLO format
            yolo_split = 'val' if split_name == 'test' else 'train'
            
            print(f"\nProcessing {split_name} split -> {yolo_split}")
            
            split_processed = 0
            
            # Get all image files
            image_files = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png")) + list(split_dir.glob("*.jpeg"))
            
            for img_file in image_files:
                # Look for corresponding label file
                label_file = split_dir / f"{img_file.stem}.txt"
                
                if not label_file.exists():
                    print(f"  WARNING: No label file found for {img_file.name}")
                    continue
                
                try:
                    # Read and convert label file
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    if not lines:
                        print(f"  WARNING: Empty label file: {label_file.name}")
                        continue
                    
                    # Convert class IDs (assuming original uses class 0 for pothole)
                    converted_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Change class ID from 0 to our pothole class (5)
                            new_class_id = self.class_mapping['pothole']
                            converted_lines.append(f"{new_class_id} {' '.join(parts[1:])}")
                    
                    if converted_lines:
                        # Copy image
                        shutil.copy2(img_file, self.output_dir / "images" / yolo_split / f"pothole_{img_file.name}")
                        
                        # Save converted labels
                        output_label_path = self.output_dir / "labels" / yolo_split / f"pothole_{img_file.stem}.txt"
                        with open(output_label_path, 'w') as f:
                            f.write('\n'.join(converted_lines))
                        
                        split_processed += 1
                        total_annotations += len(converted_lines)
                        
                        if split_processed % 100 == 0:  # Progress indicator
                            print(f"  Processed {split_processed} images...")
                
                except Exception as e:
                    print(f"  ERROR processing {img_file.name}: {e}")
                    continue
            
            print(f"  Completed {split_name}: {split_processed} images processed")
            total_processed += split_processed
        
        print(f"\nPothole dataset processing complete:")
        print(f"  Total images processed: {total_processed}")
        print(f"  Total annotations: {total_annotations}")
    
    def process_pothole_dataset(self, dataset_path, splits_file=None, split_ratio=0.8):
        """Process pothole dataset - handles both XML and YOLO formats"""
        dataset_path = Path(dataset_path)
        
        # Check if it's YOLO format (has train/test folders)
        if (dataset_path / "train").exists() and (dataset_path / "test").exists():
            print("Detected YOLO format pothole dataset")
            self.process_pothole_dataset_yolo_format(dataset_path)
        elif splits_file and Path(splits_file).exists():
            print("Detected XML format pothole dataset")
            self.process_pothole_dataset_xml_format(dataset_path, splits_file)
        else:
            print(f"ERROR: Could not determine pothole dataset format at {dataset_path}")
            print("Expected either:")
            print("  - YOLO format: train/ and test/ folders")
            print("  - XML format: annotated-images/ folder and splits.json file")
    
    def process_pothole_dataset_xml_format(self, dataset_path, splits_file):
        """Process pothole dataset (XML format) - old method renamed"""
        dataset_path = Path(dataset_path)
        
        print(f"Looking for pothole dataset at: {dataset_path.absolute()}")
        print(f"Looking for splits file at: {splits_file}")
        
        # Check if paths exist
        if not dataset_path.exists():
            print(f"ERROR: Dataset path does not exist: {dataset_path}")
            return
        
        if not Path(splits_file).exists():
            print(f"ERROR: Splits file does not exist: {splits_file}")
            return
        
        # Check annotated-images directory
        annotated_dir = dataset_path / "annotated-images"
        if not annotated_dir.exists():
            print(f"ERROR: annotated-images directory not found: {annotated_dir}")
            return
        
        print(f"Found annotated-images directory with {len(list(annotated_dir.glob('*')))} files")
        
        # Load splits
        try:
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            print(f"Loaded splits: train={len(splits.get('train', []))}, test={len(splits.get('test', []))}")
        except Exception as e:
            print(f"ERROR loading splits file: {e}")
            return
        
        total_processed = 0
        total_annotations = 0
        
        # Process each split
        for split_name, xml_files in splits.items():
            # Map 'test' to 'val' for consistency
            yolo_split = 'val' if split_name == 'test' else 'train'
            print(f"\nProcessing {split_name} split ({len(xml_files)} files) -> {yolo_split}")
            
            split_processed = 0
            
            for xml_file in xml_files:
                xml_path = dataset_path / "annotated-images" / xml_file
                img_file = xml_file.replace('.xml', '.jpg')
                img_path = dataset_path / "annotated-images" / img_file
                
                # Debug: check if files exist
                if not xml_path.exists():
                    print(f"  WARNING: XML not found: {xml_file}")
                    continue
                    
                if not img_path.exists():
                    print(f"  WARNING: Image not found: {img_file}")
                    continue
                
                try:
                    # Read image to get dimensions
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"  WARNING: Could not read image: {img_file}")
                        continue
                        
                    img_height, img_width = img.shape[:2]
                    
                    # Convert XML to YOLO format
                    yolo_annotations = self.convert_xml_to_yolo(xml_path, img_width, img_height)
                    
                    if yolo_annotations:  # Only process if there are annotations
                        # Copy image
                        shutil.copy2(img_path, self.output_dir / "images" / yolo_split / f"pothole_{img_file}")
                        
                        # Save YOLO format labels
                        label_path = self.output_dir / "labels" / yolo_split / f"pothole_{Path(img_file).stem}.txt"
                        with open(label_path, 'w') as f:
                            f.write('\n'.join(yolo_annotations))
                        
                        split_processed += 1
                        total_annotations += len(yolo_annotations)
                    else:
                        print(f"  WARNING: No annotations found in: {xml_file}")
                        
                except Exception as e:
                    print(f"  ERROR processing {xml_file}: {e}")
                    continue
            
            print(f"  Processed {split_processed} files for {split_name}")
            total_processed += split_processed
        
        print(f"\nPothole dataset processing complete:")
        print(f"  Total images processed: {total_processed}")
        print(f"  Total annotations: {total_annotations}")
    
    def create_yaml_config(self):
        """Create YAML configuration file for training"""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return yaml_path
    
    def print_dataset_statistics(self):
        """Print statistics about the merged dataset"""
        stats = {
            'train_images': len(list((self.output_dir / "images" / "train").glob("*.jpg"))),
            'val_images': len(list((self.output_dir / "images" / "val").glob("*.jpg"))),
            'train_labels': len(list((self.output_dir / "labels" / "train").glob("*.txt"))),
            'val_labels': len(list((self.output_dir / "labels" / "val").glob("*.txt")))
        }
        
        print("\n=== Dataset Statistics ===")
        print(f"Training images: {stats['train_images']}")
        print(f"Validation images: {stats['val_images']}")
        print(f"Training labels: {stats['train_labels']}")
        print(f"Validation labels: {stats['val_labels']}")
        print(f"Total classes: {len(self.class_names)}")
        print(f"Classes: {self.class_names}")
        
        # Class distribution analysis
        self.analyze_class_distribution()
        
        return stats
    
    def analyze_class_distribution(self):
        """Analyze class distribution in the dataset"""
        class_counts = {i: 0 for i in range(len(self.class_names))}
        
        for split in ['train', 'val']:
            label_dir = self.output_dir / "labels" / split
            for label_file in label_dir.glob("*.txt"):
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.strip().split()[0])
                            class_counts[class_id] += 1
        
        print("\n=== Class Distribution ===")
        for class_id, count in class_counts.items():
            if class_id < len(self.class_names):
                class_name = self.class_names[class_id]
                print(f"{class_name} (ID: {class_id}): {count} instances")

def train_yolo_model(dataset_yaml, model_size='n', epochs=100, batch_size=16, img_size=640):
    """Train YOLO model on the merged dataset"""
    try:
        from ultralytics import YOLO
        import torch
        
        # Check device availability
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"CUDA available! Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = 'cpu'
            print("CUDA not available. Using CPU training.")
            # Reduce batch size for CPU training
            if batch_size > 8:
                batch_size = 8
                print(f"Reduced batch size to {batch_size} for CPU training")
        
        # Initialize model
        model = YOLO(f'yolov8{model_size}.pt')  # n, s, m, l, x
        
        # Train the model
        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            save=True,
            plots=True,
            device=device,  # Use detected device
            project='runs/train',
            name='multi_dataset_yolo',
            workers=2 if device == 'cpu' else 8,  # Fewer workers for CPU
            patience=50,  # Early stopping
            verbose=True
        )
        
        return model, results
        
    except ImportError:
        print("Ultralytics YOLO not installed. Install with: pip install ultralytics")
        return None, None
    except Exception as e:
        print(f"Training failed: {e}")
        return None, None

# Usage example
def main():
    # Initialize processor
    processor = MultiDatasetYOLOProcessor(output_dir="merged_yolo_dataset")
    
    # Process datasets - UPDATE THESE PATHS TO YOUR ACTUAL DATASET LOCATIONS
    print("Processing Car Accident Dataset...")
    processor.process_car_accident_dataset("D:\python\citypulse\dataset\custom_dataset")
    
    print("Processing Pothole Dataset...")
    processor.process_pothole_dataset("D:\python\citypulse\dataset\dataset")  # Just the folder path, no splits.json needed
    
    # Create YAML config
    yaml_path = processor.create_yaml_config()
    print(f"Dataset configuration saved to: {yaml_path}")
    
    # Print statistics
    processor.print_dataset_statistics()
    
    # Train model
    print("\nStarting YOLO training...")
    model, results = train_yolo_model(
        yaml_path, 
        model_size='n',  # Use nano for faster CPU training
        epochs=50,       # Reduced epochs for CPU training
        batch_size=4,    # Small batch size for CPU
        img_size=416     # Smaller image size for CPU training
    )
    
    if model:
        print("Training completed successfully!")
        print(f"Best weights saved to: runs/train/multi_dataset_yolo/weights/best.pt")
    
if __name__ == "__main__":
    main()