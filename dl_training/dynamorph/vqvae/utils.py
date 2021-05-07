import torch
import torchvision.datasets as datasets


def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample


class DatasetFolderWithPaths(datasets.DatasetFolder):
    """ adapted from:
    https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d

    This method extends DatasetFolder to return the sample path+filename, which can contain important metadata
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        # original_tuple = super(DatasetFolderWithPaths, self).__getitem__(index)
        # the image file path
        # path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        # tuple_with_path = (original_tuple + (path,))
        # return tuple_with_path

        # super(DatasetFolderWithPaths, self).__getitem__(index)

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path