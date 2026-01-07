import logging
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class SummarizerHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.model = None
        self.tokenizer = None
        self.device = None

    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Модель и токенизатор успешно загружены.")
        self.initialized = True

    def preprocess(self, data):
        texts = [d.get("data") or d.get("body") for d in data]
        input_texts = []
        for text in texts:
            if isinstance(text, (bytes, bytearray)):
                text = text.decode('utf-8')
            input_texts.append("summarize: " + text)
        
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        return inputs.to(self.device)

    def inference(self, inputs):
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
        return outputs

    def postprocess(self, inference_output):
        summaries = self.tokenizer.batch_decode(inference_output, skip_special_tokens=True)
        
        return [{"summary": s} for s in summaries]