import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================
# Dataset
# ==========================
dataset_path = "PlantVillage"

IMAGE_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 5

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    dataset_path,
    transform=transform
)

# split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_data, val_data = torch.utils.data.random_split(
    dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE
)


# ==========================
# CNN Model
# ==========================

class CNN(nn.Module):
    def __init__(self, classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,16,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*14*14,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,classes)
        )


    def forward(self,x):
        x=self.features(x)
        return self.classifier(x)



model = CNN(len(dataset.classes))


# ==========================
# Training
# ==========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


for epoch in range(EPOCHS):

    model.train()

    total_loss=0

    for images,labels in train_loader:

        optimizer.zero_grad()

        output=model(images)

        loss=criterion(output,labels)

        loss.backward()

        optimizer.step()

        total_loss+=loss.item()


    print(
        f"Epoch {epoch+1}/{EPOCHS}",
        "Loss:",
        total_loss
    )


# Save model

torch.save(
    {
        "model":model.state_dict(),
        "classes":dataset.classes
    },
    "model.pth"
)


print("✅ PyTorch Model Saved")
