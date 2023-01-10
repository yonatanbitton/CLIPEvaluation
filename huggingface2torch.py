from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class HuggingFaceDataset(Dataset):
    def __init__(self, huggingface_dataset):
        self.huggingface_dataset = huggingface_dataset

    def __getitem__(self, index):
        # Get the example at the specified index
        example = self.huggingface_dataset[index]

        # Convert the example to a dictionary with the appropriate keys
        # Depending on the structure of your dataset, you may need to
        # modify this code to match the keys in your data
        data = {
            'input_ids': example['input_ids'],
            'attention_mask': example['attention_mask'],
            'labels': example['labels']
        }

        return data

    def __len__(self):
        return len(self.huggingface_dataset)


if __name__ == '__main__':
    huggingface_dataset = load_dataset("facebook/winoground", use_auth_token='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')["test"]
    # Create the dataset
    dataset = HuggingFaceDataset(huggingface_dataset)
    # Create the data loader
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print("Converted Huggingface to Torch dataset")