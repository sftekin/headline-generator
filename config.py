model_params = {
        "logging_interval": 1000,
        "saving_interval": 1000,
        "model_save_path": '.',
        "encoder_hidden_size": 512,
        "decoder_hidden_size": 512,
        "learning_rate": 0.0001,
        "padding_idx": 1,
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

