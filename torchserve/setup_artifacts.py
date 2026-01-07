import argparse
import os
import shutil
import torch
from transformers import AutoModelForSeq2SeqLM

def prepare_artifacts(model_checkpoint_path, save_dir):
    if not os.path.exists(model_checkpoint_path):
        raise FileNotFoundError(f"Папка с чекпоинтом не найдена: {model_checkpoint_path}")

    os.makedirs(save_dir, exist_ok=True)
    print(f"Подготовка артефактов в папке: {save_dir}")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint_path)
    state_dict_path = os.path.join(save_dir, "model_state_dict.pt")
    torch.save(model.state_dict(), state_dict_path)

    files_to_copy = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "spiece.model",
        "special_tokens_map.json"
    ]
    
    for filename in files_to_copy:
        src_path = os.path.join(model_checkpoint_path, filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, help="Путь к чекпоинту модели от Hugging Face.")
    parser.add_argument("--save_dir", required=True, help="Папка для сохранения артефактов для TorchServe.")
    args = parser.parse_args()
    prepare_artifacts(args.checkpoint_path, args.save_dir)
