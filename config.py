config = {
    "batch_size": 32,
    "distributed": False,
    "num_epochs": 20,              # train 20 more epochs
    "accum_iter": 10,
    "base_lr": 1.0,
    "max_padding": 72,
    "warmup": 3000,
    "file_prefix": "multi30k_model_",

    "resume_from": "multi30k_model_final.pt"
}
