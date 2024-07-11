from random import randint

from matplotlib.pyplot import imshow, axis, show
from torch import device, cuda, load, no_grad, cat
from torchvision.transforms import Compose, ToTensor

from dataset import TrainingDataset
from leads_jarvis import RRSNetwork

if __name__ == '__main__':
    device = device("cuda" if cuda.is_available() else "cpu")
    model = RRSNetwork().to(device)
    model.load_state_dict(load("leads_jarvis/checkpoints/rrs.pth"))
    model.eval()
    transform = Compose((
        ToTensor(),
    ))
    dataset = TrainingDataset("data/images", "data/masks", transform)
    with no_grad():
        image, mask = dataset[randint(0, len(dataset) - 1)]
        output = model(image.unsqueeze(0))[0]
        plot = cat((image, mask, output), 2).permute(1, 2, 0).numpy()
        imshow(plot)
        axis("off")
        show()
