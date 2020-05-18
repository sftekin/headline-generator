import sys

from config import data_params, batch_params
from load_data import LoadData
from batch_generator import BatchGenerator


def main(mode):
    dataset_path = 'data'
    data = LoadData(dataset_path=dataset_path, **data_params)

    print('Creating Batch Generator...')
    batch_creator = BatchGenerator(data_dict=data.data_dict,
                                   label_dict=data.label_dict,
                                   **batch_params)

    for x, y in batch_creator.generate('train'):
        print(x.shape, y.shape)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('no arguments given, default process of training has started')
        run_mode = 'train'
    else:
        run_mode = sys.argv[1]
    main(run_mode)
