from torch import device
from torch.cuda import is_available
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from dataset import TrainingDataset
from leads_jarvis import RRSNetwork

if __name__ == '__main__':
    transform = Compose([
        Resize((400, 400)),
        ToTensor(),
    ])
    loader = DataLoader(TrainingDataset("data/images", "data/masks", transform), batch_size=32, shuffle=True)
    device = device("cuda" if is_available() else "cpu")
    model = RRSNetwork().to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    num_epochs = 128
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(loader)}")
