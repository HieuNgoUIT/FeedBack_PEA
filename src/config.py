cfg = {
    # Model Configs
    "model": "microsoft/deberta-v3-large",
    "max_len": 512,
    
    # Train Configs
    "fold_num": 5,
    "val_fold": 0,
    "lr": 3e-6,
    "batch_size": 8,
    "valid_batch_size": 32,
    "epochs": 1, # Set to 1 because it is a demo
    "accumulation_steps": 1,
    "val_steps": 375,
    
    # GPU Optimize Settings
    "gpu_optimize_config": {
        "fp16": True,
        "freezing": True,
        "optim8bit": True,
        "gradient_checkpoint": True
    },
    
    # Path
    "input": "/kaggle/input/feedback-prize-effectiveness",
    "output": "/kaggle/working"
}
cfg = AttrDict(cfg)

LABEL_MAPPING = {"Ineffective": 0, "Adequate": 1, "Effective": 2}