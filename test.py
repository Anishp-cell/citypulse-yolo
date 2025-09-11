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
# Define a simple CNN model
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
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
    dataset_folder = r"D:\python\citypulse\merged_yolo_dataset\images\val"  # Change this to your dataset folder path
    if not os.path.exists(dataset_folder):
        print(f"Dataset folder '{dataset_folder}' not found.")
    else:
        images = [f for f in os.listdir(dataset_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if len(images) == 0:
            print("No images found in dataset folder.")
        else:
            print("\nAvailable images:")
            for i, img_name in enumerate(images):
                print(f"{i+1}. {img_name}")
            img_choice = int(input("Choose image number: ")) - 1
            if 0 <= img_choice < len(images):
                image_path = os.path.join(dataset_folder, images[img_choice])
                test_image(image_path)
            else:
                print("Invalid choice.")
elif choice == "2":
    test_camera()
else:
    print("Invalid choice.")
