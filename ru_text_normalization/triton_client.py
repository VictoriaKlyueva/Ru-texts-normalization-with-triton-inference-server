import tritonclient.http as httpclient
from transformers import GPT2Tokenizer
import numpy as np

from ru_text_normalization.constants.constants import CLIENT_URL, HF_MODEL_NAME
from ru_text_normalization.utils.logger import logger
from ru_text_normalization.utils.postprocessing import TextPreprocessor, TextPostprocessor


class TritonClient:
    """
    Client for interacting with Triton Inference Server for T5 text normalization.

    Attributes:
        url (str): Triton server URL
        client (httpclient.InferenceServerClient): HTTP client for Triton
        preprocessor (TextPreprocessor): Text preprocessor
        postprocessor (TextPostprocessor): Text postprocessor
        tokenizer (AutoTokenizer): Tokenizer for text processing
    """

    def __init__(self, url: str = CLIENT_URL, model_name: str = HF_MODEL_NAME):
        """
        Initialize Triton client.

        Args:
            url (str): Triton server URL
            model_name (str): Name of the model to use for tokenization
        """

        self.url = url
        self.model_name = model_name

        self.preprocessor = TextPreprocessor()
        self.postprocessor = TextPostprocessor()

        self.max_input_length = 256
        self.max_output_length = 256

        self.client = httpclient.InferenceServerClient(url=url)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, eos_token='</s>')

        logger.info(f"Triton client initialized with URL: {url}")

    def normalize_text(self, text: str):
        """
        Normalize text using Triton Inference Server with finetuned FRED model.

        Args:
            text (str): Input text for normalization

        Returns:
            str: Normalized text
        """
        logger.info(f"Starting text normalization for text: {text}")

        preprocessed_text = self.preprocessor.preprocess_sentence(text)

        inputs = self.tokenizer(
            preprocessed_text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length
        )

        input_ids = httpclient.InferInput("input_ids", inputs["input_ids"].shape, "INT64")
        attention_mask = httpclient.InferInput("attention_mask", inputs["attention_mask"].shape, "INT64")

        input_ids.set_data_from_numpy(inputs["input_ids"].astype(np.int64))
        attention_mask.set_data_from_numpy(inputs["attention_mask"].astype(np.int64))

        response = self.client.infer(
            model_name="text_normalization",
            inputs=[input_ids, attention_mask],
            outputs=[httpclient.InferRequestedOutput("output_ids")]
        )

        output_ids = response.as_numpy("output_ids")
        output_text = self.tokenizer.decode(output_ids[0][1:], skip_special_tokens=True)

        normalized_text = self.postprocessor.postprocess_outputs(text, output_text)

        return normalized_text
