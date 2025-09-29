import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os

# -------------------------
# Check CUDA availability
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"CUDA Version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Running on CPU.")

# -------------------------
# Simple CNN (size-agnostic via GAP)
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.gap   = nn.AdaptiveAvgPool2d((1, 1))  # global average pooling
        self.fc1   = nn.Linear(32, 10)             # 32 channels -> 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool(x)  # 1/2 size
        x = F.relu(self.conv2(x)); x = self.pool(x)  # 1/4 size
        x = self.gap(x)                               # -> [B, 32, 1, 1]
        x = torch.flatten(x, 1)                       # -> [B, 32]
        x = self.fc1(x)
        return x

model = SimpleCNN().to(device)
print("Model loaded on:", next(model.parameters()).device)

# -------------------------
# Transform for image
# -------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# -------------------------
# Function to test image
# -------------------------
def test_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
    print(f"Inference done on image: {image_path}")
    print(f"Output tensor shape: {output.shape}")
    print(f"First 5 values: {output[0][:5]}")

# -------------------------
# Function to test live camera
# -------------------------
def test_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera.")
        return
    print("Press 'q' to capture and run inference.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Camera Feed - Press q to capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
            print("Inference done on captured frame.")
            print(f"Output tensor shape: {output.shape}")
            print(f"First 5 values: {output[0][:5]}")
            break
    cap.release()
    cv2.destroyAllWindows()

# -------------------------
# Choose Mode
# -------------------------
print("\nChoose mode:")
print("1. Image from dataset")
print("2. Live camera")
choice = input("Enter 1 or 2: ")

if choice == "1":
    dataset_folder = r"D:\python\citypulse\merged_yolo_dataset\images\val"  # Change if needed
    if not os.path.exists(dataset_folder):
        print(f"Dataset folder '{dataset_folder}' not found.")
    else:
        images = [f for f in os.listdir(dataset_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if len(images) == 0:
            print("No images found in dataset folder.")
        else:
            print("\nAvailable images:")
            for i, img_name in enumerate(images):
                print(f"{i+1}. {img_name}")
            try:
                img_choice = int(input("Choose image number: ")) - 1
                if 0 <= img_choice < len(images):
                    image_path = os.path.join(dataset_folder, images[img_choice])
                    test_image(image_path)
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Please enter a valid number.")
elif choice == "2":
    test_camera()
else:
    print("Invalid choice.")
