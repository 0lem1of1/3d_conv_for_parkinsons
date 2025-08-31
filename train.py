

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path

class PDGaitAnalysisSystem(nn.Module):
    """
    A 3D CNN model for classifying Parkinson's Disease from gait videos.
    The architecture is designed to capture both spatial and temporal features.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        self.flattened_size = 0 
        
        self.fc = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1) 
        return self.fc(x)

class ParkinsonRGBDataset(Dataset):
    """
    Custom PyTorch Dataset for loading video frames for gait analysis.
    It expects a specific directory structure and labels data based on folder names.
    """
    def __init__(self, root, clip_length=32, transform=None):
        self.clip_length = clip_length
        self.samples = []
        self.transform = transform

        root_path = Path(root)
        for subject_dir in root_path.iterdir():
            if subject_dir.is_dir():
                label = 1 if "_PG" in subject_dir.name else 0
                frames_dir = subject_dir / "frames"
                if frames_dir.exists():
                    for video_dir in frames_dir.iterdir():
                        if video_dir.is_dir():
                            frame_files = sorted(video_dir.glob("*.png"))
                            if len(frame_files) >= clip_length:
                                self.samples.append((video_dir, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, label = self.samples[idx]
        frame_files = sorted(video_dir.glob("*.png"))

        total_frames = len(frame_files)
        indices = torch.linspace(0, total_frames - 1, self.clip_length).long()

        frames = []
        for i in indices:
            frame = Image.open(frame_files[i]).convert('RGB')
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        video_tensor = torch.stack(frames)
        video_tensor = video_tensor.permute(1, 0, 2, 3)

        return video_tensor, label

def main():
    """
    Main function to set up and run the training process.
    """
    base_dataset_dir = "/path/to/dataset" 
    num_epochs = 20
    batch_size = 4
    learning_rate = 0.001
    clip_length = 32 

    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading dataset...")
    full_dataset = ParkinsonRGBDataset(base_dataset_dir, clip_length=clip_length, transform=transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Dataset loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = PDGaitAnalysisSystem(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total_train += target.size(0)
            correct_train += predicted.eq(target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        train_acc = 100. * correct_train / total_train

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        val_acc = 100. * val_correct / val_total
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved with accuracy: {val_acc:.2f}%")
            
    print("Training finished.")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
