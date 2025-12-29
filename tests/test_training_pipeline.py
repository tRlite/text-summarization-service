import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import glob
import os
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from src import train, run_evaluation

@pytest.fixture
def fake_dataset(mocker):
    fake_data = {
        'text': ['Пример текста для тестов 1.', 'Пример текста для тестов 2.'],
        'summary': ['Саммари 1.', 'Саммари 2.']
    }
    dataset = Dataset.from_dict(fake_data)

    dataset_dict = DatasetDict({'train': dataset, 'test': dataset})

    mocker.patch('src.data_loader.load_dataset', return_value=dataset_dict)
    mocker.patch('src.run_evaluation.load_dataset', return_value=dataset)


def test_training_smoke(test_config, temp_output_dir, fake_dataset):
    train.main("tests/test_config.yaml")
    
    assert os.path.isdir(temp_output_dir)
    
    checkpoint_dirs = glob.glob(os.path.join(temp_output_dir, "checkpoint-*"))
    assert len(checkpoint_dirs) > 0, "Обучение не создало контрольную точку"
    
    checkpoint_path = checkpoint_dirs[0]
    bin_file_exists = os.path.isfile(os.path.join(checkpoint_path, "pytorch_model.bin"))
    safetensors_file_exists = os.path.isfile(os.path.join(checkpoint_path, "model.safetensors"))

    assert bin_file_exists or safetensors_file_exists, "Файл с весами модели (bin или safetensors) не был сохранен"


def test_evaluation_finetuned_smoke(test_config, temp_output_dir, fake_dataset):
    train.main("tests/test_config.yaml")
    model_path = glob.glob(os.path.join(temp_output_dir, "checkpoint-*"))[0]
    
    run_evaluation.main(
        config_path="tests/test_config.yaml",
        model_path=model_path
    )


def test_evaluation_zeroshot_smoke(test_config, fake_dataset):
    run_evaluation.main(config_path="tests/test_config.yaml")

def test_compute_metrics_logic(mocker):
    class MockTokenizer:
        def batch_decode(self, ids, skip_special_tokens):
            decoded_sequences = []
            for seq in ids:
                if skip_special_tokens:
                    filtered_seq = [str(token) for token in seq if token != -100 and token != self.pad_token_id]
                    decoded_sequences.append(" ".join(filtered_seq))
                else:
                    decoded_sequences.append(" ".join(map(str, seq)))
            return decoded_sequences
        @property
        def pad_token_id(self):
            return 0


    from src.model import SummarizationTrainer
    trainer_module = SummarizationTrainer({'model': {}, 'training': {}})
    trainer_module.tokenizer = MockTokenizer()
    
    predictions = [[1, 2, 3], [4, 5, 0]]
    labels = [[1, 2, 3], [4, 5, -100]]
    eval_pred = (predictions, labels)

    class MockRouge:
        def compute(self, predictions, references, use_stemmer):
            assert predictions == ['1 2 3', '4 5']
            assert references == ['1 2 3', '4 5']
            return {"rouge1": 1.0, "rougeL": 1.0}

    mocker.patch('evaluate.load', return_value=MockRouge())

    metrics = trainer_module._compute_metrics(eval_pred)

    assert "rouge1" in metrics
    assert "gen_len" in metrics
    assert metrics["rouge1"] == 1.0
