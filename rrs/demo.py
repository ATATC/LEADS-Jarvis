from random import randint
from time import time

from leads_jarvis import RRSNetwork
from matplotlib.pyplot import imshow, axis, show
from torch import device, cuda, load, no_grad, cat
from torchvision.transforms import Compose, Resize, ToTensor

from dataset import TrainingDataset

if __name__ == "__main__":
    device = device("cuda" if cuda.is_available() else "cpu")
    model = RRSNetwork().to(device)
    model.load_state_dict(load("leads_jarvis/checkpoints/rrs.pth"))
    model.eval()
    transform = Compose((
        Resize(400),
        ToTensor(),
    ))
    dataset = TrainingDataset("data/images", "data/masks", transform)
    with no_grad():
        image, mask = dataset[randint(0, len(dataset) - 1)]
        start = time()
        output = model(image.unsqueeze(0))[0]
        print(f"{(time() - start) * 1000} MS")
        plot = cat((image, mask, output), 2).permute(1, 2, 0).numpy()
        imshow(plot)
        axis("off")
        show()
