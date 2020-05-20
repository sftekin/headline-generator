import sys
import pickle as pkl

from config import data_params, batch_params, model_params, train_params
from load_data import LoadData
from batch_generator import BatchGenerator
from trainer import train
from tester import test


def main(mode):
    dataset_path = 'data'
    data = LoadData(dataset_path=dataset_path, **data_params)

    print('Creating Batch Generator...')
    batch_gen = BatchGenerator(data_dict=data.data_dict,
                               label_dict=data.label_dict,
                               **batch_params)

    if mode == 'train':
        train(vocabs=[data.word2int, data.int2word],
              batch_gen=batch_gen,
              train_params=train_params,
              model_params=model_params)

    elif mode == 'test':
        print('Loading model')
        model_file = open('results/seq2seq.pkl', 'rb')
        model = pkl.load(model_file)
        print('Testing model...')
        test(model, data.int2word, batch_gen)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('no arguments given, default process of training has started')
        run_mode = 'train'
    else:
        run_mode = sys.argv[1]
    main(run_mode)
