from os.path import exists

from leads_jarvis import RRSNetwork
from rich.progress import Progress as _Progress
from torch import device, save, load
from torch.cuda import is_available
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomRotation

from dataset import TrainingDataset

if __name__ == "__main__":
    transform = Compose((
        RandomHorizontalFlip(),
        RandomRotation(10),
        ToTensor(),
    ))
    loader = DataLoader(TrainingDataset("data/images", "data/masks", transform), batch_size=16, shuffle=True)
    device = device("cuda" if is_available() else "cpu")
    model = RRSNetwork().to(device)
    if exists("leads_jarvis/checkpoints/rrs.pth"):
        model.load_state_dict(load("leads_jarvis/checkpoints/rrs.pth"))
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    num_epochs = 1024
    with _Progress(refresh_per_second=True) as progress:
        task = progress.add_task("[white]Training...", total=num_epochs)
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
            save(model.state_dict(), "leads_jarvis/checkpoints/rrs.pth")
            progress.update(task, advance=1, description=f"[white]Training loss: {epoch_loss / len(loader):.3f}")
