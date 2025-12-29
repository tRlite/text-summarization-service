import logging
import evaluate
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

logger = logging.getLogger(__name__)

class SummarizationTrainer:
    def __init__(self, config):
        self.model_config = config['model']
        self.training_config = config['training']
        self.tokenizer = None
        self.model = None

    def _load_model_and_tokenizer(self):
        logger.info(f"Загрузка модели и токенизатора: {self.model_config['name']}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config['name'])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_config['name'])

    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
       
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        rouge = evaluate.load("rouge")
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def train(self, train_dataset, eval_dataset):
        self._load_model_and_tokenizer()

        training_args = Seq2SeqTrainingArguments(
            **self.training_config
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )

        logger.info("Начало обучения модели...")
        trainer.train()
    
        trainer.save_model() 
        logger.info(f"Модель сохранена в: {self.training_config['output_dir']}")

