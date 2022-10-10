import numpy as np
import torch
import pandas as pd
from read_data import TestDataset
from pathlib import Path
from torch.utils.data import DataLoader


def check_accuracy_test(test_loader, model, data_path):
    predictions = np.array([])
    model.eval()
    with torch.no_grad():
        for image in test_loader:
            scores = model(image)
            _, predicted = torch.max(scores, 1)
            predictions = np.append(predictions, predicted)
    model.train()
    test_df = pd.read_csv(data_path / 'sample_submission.csv')
    del test_df['Label']
    test_df['Label'] = predictions.astype(int)
    test_df.to_csv(data_path / 'sample_submission.csv', index=False)


if __name__ == '__main__':
    data_path = Path().cwd() / 'dataset'
    test_dataset = TestDataset(data_path / 'test.csv')
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    model = torch.load(Path().cwd() / 'model' / 'CNN-model.pt')
    check_accuracy_test(test_loader, model, data_path)