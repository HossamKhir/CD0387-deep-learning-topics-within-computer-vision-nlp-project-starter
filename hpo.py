# TODO: Import your dependencies.
# For instance, below are some dependencies you might need if you are using Pytorch
# import numpy as np
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def test(model, test_loader):
def test(model, test_loader, criterion):
    """
    TODO: Complete this function that can take a model and a
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    """
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for features, label in test_loader:
            features = features.to(DEVICE)
            label = label.to(DEVICE)

            out = model(features)
            test_loss += criterion(out, label).sum().item()
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    n = len(test_loader.dataset)
    test_loss /= n
    print(f"Average loss: {test_loss:.4f}, Accuracy: {correct}/{n}")


def train(model, train_loader, criterion, optimizer):
    """
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    """
    model.train()
    for features, label in train_loader:
        features = features.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()
        out = model(features)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

    return model


def net():
    """
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    """
    # model = models.resnet50(pretrained=True)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(512, 133),
        nn.LogSoftmax(dim=1),
    )

    return model


def create_data_loaders(data, batch_size):
    """
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    """
    return torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True
    )


def main(args):
    """
    TODO: Initialize a model by calling the net function
    """
    model = net()
    model.to(DEVICE)

    """
    TODO: Create your loss and optimizer
    """
    loss_criterion = nn.NLLLoss()
    optimizer = optim.Adam(
        model.fc.parameters(), lr=args.lr, betas=(args.beta1, args.beta2)
    )

    """
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    set_train = torchvision.datasets.ImageFolder(
        args.train_dir, transform=transform_train
    )
    set_valid = torchvision.datasets.ImageFolder(
        args.valid_dir, transform=transform_test
    )

    train_loader = create_data_loaders(
        data=set_train, batch_size=args.batch_size
    )
    test_loader = create_data_loaders(
        data=set_valid, batch_size=args.test_batch_size
    )

    for _ in range(args.epochs):
        model = train(model, train_loader, loss_criterion, optimizer)

        """
        TODO: Test the model to see its accuracy
        """
        test(model, test_loader, loss_criterion)

    """
    TODO: Save the trained model
    """
    path = os.path.join(os.environ["SM_MODEL_DIR"], "model.pth")
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """
    TODO: Specify all the hyperparameters you need to use to train your model.
    """

    parser.add_argument("--epochs", default=4, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--test-batch-size", default=128, type=int)
    parser.add_argument("--lr", default=1.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument(
        "--train-dir", default=os.environ["SM_CHANNEL_TRAINING"], type=str
    )
    parser.add_argument(
        "--valid-dir", default=os.environ["SM_CHANNEL_VALIDATION"], type=str
    )

    args = parser.parse_args()

    main(args)
