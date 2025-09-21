import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset

def load_data(batch_size=64, classes_to_use=[0, 1, 2]):
    """
    Load CIFAR-10 dataset but only keep selected classes.
    Example: [0,1,2] = airplane, automobile, bird
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = datasets.CIFAR10(root='./data/raw', train=True,
                                download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data/raw', train=False,
                               download=True, transform=transform)

    # Filter dataset for given classes
    train_idx = [i for i, (_, label) in enumerate(trainset) if label in classes_to_use]
    test_idx = [i for i, (_, label) in enumerate(testset) if label in classes_to_use]

    train_subset = Subset(trainset, train_idx)
    test_subset = Subset(testset, test_idx)

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader
