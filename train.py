import numpy as np

from read_data import TrainDataset
from torch.utils.data import DataLoader
import torch
from model import CNN
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channel = 1
num_classes = 10
learning_rate = 0.001
batch_size = 32
num_epochs = 8


# Load Dataset
train_dataset = TrainDataset(r'C:\Users\marci\Desktop\Python\Digit-Recognizer\dataset\train.csv')
train_loader =  DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

#test_dataset = TestDataset(r'C:\Users\marci\Desktop\Python\Digit-Recognizer\dataset\test.csv')
#test_loader =  DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = CNN(in_channel=in_channel, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # CUDA
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f" {num_correct} / {num_samples} {(float(num_correct/num_samples))*100:.2f}")
    model.train()
    # Add model eval
    model.eval()


check_accuracy(loader=train_loader, model=model)

images = pd.read_csv(r'C:\Users\marci\Desktop\Python\Digit-Recognizer\dataset\test.csv')
images_values = images.values.reshape(len(images),1,28,28)
predictions = []
with torch.no_grad():
    for i in range(len(images_values)):
        array = np.array([images.values.reshape(len(images),1,28,28)[i]])
        tensor = torch.tensor(array,dtype=torch.float32)
        scores = model(tensor)
        _, predicted = torch.max(scores, 1)
        predictions.append(predicted.item())

print(len(predictions))
df = pd.read_csv(r'C:\Users\marci\Desktop\Python\Digit-Recognizer\dataset\sample_submission.csv')
del df['Label']
df['Label'] = predictions
df.to_csv(r'C:\Users\marci\Desktop\Python\Digit-Recognizer\dataset\sample_submission.csv',index=False)