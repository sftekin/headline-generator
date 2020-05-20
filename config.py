model_params = {
    'encoder_hidden_dim': 512,
    'decoder_hidden_dim': 512,
    'encoder_num_layer': 2,
    'decoder_num_layer': 2,
    'dropout_prob': 0.2,
    }

train_params = {
    'num_epoch': 50,
    'learn_rate': 0.01,
    'tf_ratio': 1,
    'clip': 1,
    'eval_every': 200
}

data_params = {
        "content_len": 50,
        "title_len": 15,
        "num_samples": -1,
        "num_sentence": 3,
        "test_ratio": 0.1,
        "val_ratio": 0.1,
        "shuffle": True,
        "unk_threshold": 15
    }

batch_params = {
    'batch_size': 256,
    'num_works': 0,
    'shuffle': True,
}

