import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)

class SummarizationDataModule:
    def __init__(self, config, tokenizer):
        self.config = config['data']
        self.tokenizer = tokenizer
        self.prefix = "summarize: "

    def _load_and_split_dataset(self):
        logger.info(f"Загрузка датасета: {self.config['dataset_name']}")
        dataset = load_dataset(
            self.config['dataset_name'],
            split={
                'train': self.config['train_split'],
                'test': self.config['test_split']
            }
        )
        logger.info(f"Размер обучающей выборки: {len(dataset['train'])}")
        logger.info(f"Размер тестовой выборки: {len(dataset['test'])}")
        return dataset

    def _preprocess_function(self, examples):
        inputs = [self.prefix + doc for doc in examples[self.config['text_column']]]
        model_inputs = self.tokenizer(
            inputs, 
            max_length=self.config['max_source_length'], 
            truncation=True
        )

        labels = self.tokenizer(
            text_target=examples[self.config['summary_column']], 
            max_length=self.config['max_target_length'], 
            truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def prepare_data(self):
        dataset = self._load_and_split_dataset()
        logger.info("Токенизация данных...")
        tokenized_datasets = dataset.map(
            self._preprocess_function, 
            batched=True, 
            remove_columns=[self.config['text_column'], self.config['summary_column']]
        )
        return tokenized_datasets['train'], tokenized_datasets['test']

