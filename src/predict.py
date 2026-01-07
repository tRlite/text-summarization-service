import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def load_model(model_dir, device):
    print(f"Загрузка модели из: {model_dir}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def summarize_texts(texts, model, tokenizer, device, max_length=256, num_beams=4):
    summaries = []
    prefix = "summarize: "
    
    for text in tqdm(texts):
        input_text = prefix + text
        input_ids = tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512,
            truncation=True
        ).input_ids.to(device)

        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )[0]
        
        summary = tokenizer.decode(output_ids, skip_special_tokens=True)
        summaries.append(summary)
        
    return summaries

def main(input_path, output_path, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Папка с моделью не найдена по пути: {model_path}")
    model, tokenizer = load_model(model_path, device)

    print(f"Чтение данных из: {input_path}")
    try:
        df_input = pd.read_csv(input_path)
        if 'text' not in df_input.columns:
            raise ValueError("Входной CSV файл должен содержать колонку 'text'")
        texts_to_summarize = df_input['text'].tolist()
    except Exception as e:
        print(f"Ошибка при чтении входного файла: {e}")
        return

    generated_summaries = summarize_texts(texts_to_summarize, model, tokenizer, device)

    df_output = pd.DataFrame({
        'original_text': texts_to_summarize,
        'summary': generated_summaries
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Сохранение результатов в: {output_path}")
    df_output.to_csv(output_path, index=False)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Офлайн-инференс для модели суммаризации.")
    parser.add_argument(
        "--input_path", 
        type=str, 
        required=True, 
        help="Путь к входному CSV файлу с колонкой 'text'."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True, 
        help="Путь для сохранения выходного CSV файла с результатами."
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="./model", 
        help="Путь к папке с обученной моделью."
    )
    
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.model_path)
