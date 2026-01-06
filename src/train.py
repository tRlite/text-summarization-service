import argparse
import logging
import yaml
import torch
import numpy as np
import random

from .data_loader import SummarizationDataModule
from .model import SummarizationTrainer

import mlflow
import mlflow.transformers


def setup_logging(config):
    logging.basicConfig(
        level=config['logging']['level'],
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(config['logging']['log_file']),
            logging.StreamHandler()
        ]
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    setup_logging(config)
    logger = logging.getLogger(__name__)

    mlflow.set_experiment(config['project_name'])
    mlflow.transformers.autolog(log_models=True, disable=False)

    with mlflow.start_run() as run:
    
        logger.info(f"Запуcк MLflow Run ID: {run.info.run_id}")
        mlflow.log_params(config)
        mlflow.log_artifact(config_path, "configs")
        mlflow.log_artifact("dvc.lock", "dvc")

        set_seed(config['random_seed'])
        logger.info(f"Seed установлен на: {config['random_seed']}")

        trainer_module = SummarizationTrainer(config)
        #trainer_module._load_model_and_tokenizer()
        tokenizer = trainer_module._load_tokenizer_only()
        
        data_module = SummarizationDataModule(config, tokenizer)
        train_dataset, eval_dataset = data_module.prepare_data()

        trainer_module.train(train_dataset, eval_dataset)
        
        logger.info("Пайплайн успешно завершен.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Запуск обучения модели суммаризации.")
    parser.add_argument("config", type=str, help="Путь к конфигурационному файлу")
    
    args = parser.parse_args()
    main(args.config)

