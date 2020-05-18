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

