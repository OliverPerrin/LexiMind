# scripts/train.py
from src.training.trainer import Trainer
from src.utils.config import load_config

if __name__ == "__main__":
    config = load_config("configs/training/default.yaml")
    trainer = Trainer(config)
    trainer.train()