import argparse
import logging
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import evaluate

def setup_logging(config):
    logging.basicConfig(
        level=config['logging']['level'],
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(config['logging']['log_file']),
            logging.StreamHandler()
        ]
    )

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model_and_tokenizer(model_path, device, logger):
    logger.info(f"Загрузка модели из: {model_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def load_evaluation_data(config, logger):
    data_config = config['data']
    logger.info(f"Загрузка данных для оценки: {data_config['dataset_name']}, сплит: {data_config['test_split']}")
    dataset = load_dataset(data_config['dataset_name'], split=data_config['test_split'])
    return dataset

def generate_summaries(dataset, model, tokenizer, config, device, logger):
    data_config = config['data']
    predictions = []
    references = []
    
    logger.info("Начало генерации саммари для тестового набора...")
    for example in tqdm(dataset):
        raw_text = example[data_config['text_column']]
        prefix = "summarize: "
        input_text = prefix + raw_text

        input_ids = tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=data_config['max_source_length'], 
            truncation=True
        ).input_ids.to(device)

        output_ids = model.generate(
            input_ids,
            max_length=data_config['max_target_length'],
            num_beams=4,
            early_stopping=True
        )[0]
        
        summary = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        predictions.append(summary)
        references.append(example[data_config['summary_column']])
        
    return predictions, references

def calculate_metrics(predictions, references, logger):
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    
    logger.info("--- Итоговые метрики ---")
    for key, value in results.items():
        logger.info(f"{key}: {value*100:.2f}")
    
    return results

def show_examples(dataset, predictions, references, config, logger, num_examples=5):
    data_config = config['data']
    logger.info("\n--- Примеры генерации ---")

    num_examples = min(num_examples, len(dataset))
    
    df = pd.DataFrame({
        'Исходный текст': [ex[data_config['text_column']] for ex in dataset.select(range(num_examples))],
        'Эталонное саммари': references[:num_examples],
        'Сгенерированное саммари': predictions[:num_examples]
    })
    
    pd.set_option('display.max_colwidth', None)
    print(df.to_string())


def main(config_path, model_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config(config_path)

    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Используемое устройство: {device}")

    if model_path:
        model_identifier = model_path
    else:
        model_identifier = config['model']['name']
        logger.warning("Путь к модели не указан. Запуск оценки базовой модели.")

    model, tokenizer = load_model_and_tokenizer(model_identifier, device, logger)
    eval_dataset = load_evaluation_data(config, logger)
    
    predictions, references = generate_summaries(eval_dataset, model, tokenizer, config, device, logger)
    calculate_metrics(predictions, references, logger)
    show_examples(eval_dataset, predictions, references, config, logger)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Оценка обученной модели суммаризации.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        help="Путь к папке с обученной моделью",
        default=None
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        required=True, 
        help="Путь к конфигурационному файлу, использованному для обучения"
    )
    
    args = parser.parse_args()
    main(args.config_path, args.model_path)