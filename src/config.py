from attrdict import AttrDict

cfg = {
    # Model Configs
    "model": "microsoft/deberta-v3-large",
    "max_len": 1024,
    #"fold" : 0,
    # Train Configs
    "fold_num": 5,
    #"val_fold": 0,
    "lr": 3e-6,
    "batch_size": 8,
    "valid_batch_size": 32,
    "epochs": 1, # Set to 1 because it is a demo
    "accumulation_steps": 1,
    "val_steps": 1500,
    
    # GPU Optimize Settings
    "gpu_optimize_config": {
        "fp16": True,
        "freezing": True,
        "optim8bit": True,
        "gradient_checkpoint": True
    },
    
    # Path
    "input": "input/feedback-prize-effectiveness",
    "output": "output"
}

cfg = AttrDict(cfg)