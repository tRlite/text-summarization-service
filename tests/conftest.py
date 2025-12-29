import pytest
import yaml
import shutil
import os
from transformers import AutoTokenizer

TEST_CONFIG_PATH = "tests/test_config.yaml"

@pytest.fixture(scope="session")
def test_config():
    with open(TEST_CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return config

@pytest.fixture(scope="session")
def tokenizer(test_config):
    return AutoTokenizer.from_pretrained(test_config['model']['name'])

@pytest.fixture(scope="function")
def temp_output_dir(test_config):
    path = test_config['training']['output_dir']
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    
    yield path
    
    shutil.rmtree(path)
