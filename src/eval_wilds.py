import random
from copy import deepcopy

import torch
import torchvision.transforms as transforms
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader

dataset = get_dataset(dataset="iwildcam", download=True)


# Get the test set
test_data = dataset.get_subset(
    "test",
    transform=transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    ),
)

def main():
    # Prepare the evaluation data loader
    test_loader = get_eval_loader("standard", test_data, batch_size=16)

    all_y_true = torch.tensor([])
    all_y_pred = torch.tensor([])
    all_metadata = []
    # Get predictions for the full test set
    for idx, (x, y_true, metadata) in enumerate(test_loader):
        if idx > 10:
            break
        y_pred = deepcopy(y_true)
        if random.random() < 0.5:
            for i in range(len(y_pred)):
                if random.random() < 0.5:
                    y_pred[i] += 1
        all_y_true = torch.cat((all_y_true, y_true), dim=0)
        all_y_pred = torch.cat((all_y_pred, y_pred), dim=0)
        all_metadata.append(metadata)

    eval_score = dataset.eval(all_y_pred, all_y_true, all_metadata)
    print(f'eval_score: {eval_score}')

    print("Done")

if __name__ == '__main__':
    main()
