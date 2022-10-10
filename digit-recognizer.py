import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from train import check_accuracy_train, train, plot_loss_curve


def initParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('learning_rate', help="Number of scenario", type=float)
    parser.add_argument('epochs', help="Number of scenario", type=int)

    return parser


if __name__ == "__main__":
    parser = initParser()
    args = parser.parse_args()

    learning_rate = args.learning_rate
    num_epochs = args.epochs

    model, device, train_loader, losses = train(num_epochs=num_epochs, learning_rate=learning_rate)
    torch.save(model, Path().cwd() / 'model' / 'CNN-model.pt')
    check_accuracy_train(loader=train_loader, model=model)
    plot_loss_curve(losses=losses, num_epochs=num_epochs, learning_rate=learning_rate)