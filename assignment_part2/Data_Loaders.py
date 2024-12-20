import sklearn.model_selection
import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/submission.csv', delimiter=',')
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data
        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        # np.savetxt('normalized.csv', self.normalized_data, delimiter=',', fmt='%f')
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
# STUDENTS: __len__() returns the length of the dataset
        return len(self.normalized_data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = self.normalized_data[idx]
            input = np.array(idx[:-1], dtype=np.float32)
            label = np.array(idx[-1], dtype=np.float32)
            return {'input': input, 'label': label}
# STUDENTS: for this example, __getitem__() mu  st return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        self.train_loader, self.test_loader = sklearn.model_selection.train_test_split(self.nav_dataset, test_size=0.25)

# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # print(len(data_loaders.test_loader))

    data = data_loaders.nav_dataset.data
    npdat = np.array(data)
    uniquevalues, cnt = np.unique(npdat[:,-1], return_counts=True)
    print(uniquevalues)
    print(cnt)

    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']

    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
