import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.data_loader import SummarizationDataModule

@pytest.fixture
def data_module(test_config, tokenizer):
    return SummarizationDataModule(test_config, tokenizer)


@pytest.mark.parametrize(
    "input_text, input_summary, expected_prefix",
    [
        ("Это обычный текст для теста.", "Обычное саммари.", "summarize: "),
        
        ("", "Непустое саммари.", "summarize: "),
        
        ("Непустой текст.", "", "summarize: "),
        
        ("", "", "summarize: "),
    ],
    ids=["standard_case", "empty_text", "empty_summary", "both_empty"]
)
def test_preprocess_basic_cases(data_module, tokenizer, input_text, input_summary, expected_prefix):
    fake_data = {'text': [input_text], 'summary': [input_summary]}
    processed = data_module._preprocess_function(fake_data)

    decoded_input = tokenizer.decode(processed['input_ids'][0], skip_special_tokens=True)
    decoded_label = tokenizer.decode(processed['labels'][0], skip_special_tokens=True)
    
    assert decoded_input.startswith(expected_prefix.strip())
    assert decoded_label == input_summary

def test_preprocess_truncation(data_module, test_config, tokenizer):
    max_len_source = test_config['data']['max_source_length']
    max_len_target = test_config['data']['max_target_length']
    
    long_text = "слово " * (max_len_source + 20)
    long_summary = "саммари " * (max_len_target + 20)
    
    fake_data = {'text': [long_text], 'summary': [long_summary]}
    processed = data_module._preprocess_function(fake_data)

    assert len(processed['input_ids'][0]) == max_len_source
    assert len(processed['labels'][0]) == max_len_target

    assert processed['input_ids'][0][-1] == tokenizer.eos_token_id
    assert processed['labels'][0][-1] == tokenizer.eos_token_id

