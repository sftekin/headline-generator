from dataset import HeadlineDataset
from torch.utils.data import DataLoader


class BatchGenerator:
    def __init__(self, data_dict, label_dict, **params):
        self.data_dict = data_dict
        self.label_dict = label_dict

        self.batch_size = params.get('batch_size', 16)
        self.num_works = params.get('num_works', 4)
        self.shuffle = params.get('shuffle', True)

        self.dataset_dict, self.dataloader_dict = self.__create_data()

    def generate(self, data_type):
        """
        :param data_type: can be 'test', 'train' and 'validation'
        :return: img tensor, label numpy_array
        """
        selected_loader = self.dataloader_dict[data_type]
        yield from selected_loader

    def __create_data(self):

        im_dataset = {}
        for i in ['test', 'train', 'validation']:
            im_dataset[i] = HeadlineDataset(articles=self.data_dict[i],
                                            titles=self.label_dict[i])

        im_loader = {}
        for i in ['test', 'train', 'validation']:
            im_loader[i] = DataLoader(im_dataset[i],
                                      batch_size=self.batch_size,
                                      shuffle=self.shuffle,
                                      num_workers=self.num_works,
                                      drop_last=True)
        return im_dataset, im_loader
