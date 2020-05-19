model_params = {
    'encoder_hidden_dim': 512,
    'decoder_hidden_dim': 512,
    'encoder_num_layer': 1,
    'decoder_num_layer': 1,
    'dropout_prob': 0.2,
    }

train_params = {
    'num_epoch': 50,
    'learn_rate': 0.0003,
    'tf_ratio': 1,
    'clip': 5,
    'eval_every': 100
}

data_params = {
        "content_len": 50,
        "title_len": 15,
        "num_samples": 100,
        "num_sentence": 3,
        "test_ratio": 0.1,
        "val_ratio": 0.1,
        "shuffle": True,
        "unk_threshold": 5
    }

batch_params = {
    'batch_size': 16,
    'num_works': 0,
    'shuffle': True,
    'use_transform': True,
    "input_size": (224, 224)
}

