import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "1.0.0"

from .data_loader import SummarizationDataModule
from .model import SummarizationTrainer
