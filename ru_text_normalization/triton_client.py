import tritonclient.http as httpclient
from transformers import GPT2Tokenizer
import numpy as np
import time

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

        logger.info(f"Initializing Triton client with URL: {url}")
        try:
            self.client = httpclient.InferenceServerClient(
                url=url,
                verbose=True,  # Включаем подробное логирование
                ssl=False,
                connection_timeout=120.0,  # Увеличиваем таймаут
                network_timeout=120.0
            )
            logger.info("Successfully created InferenceServerClient")
            
            # Проверяем доступность сервера
            if not self.client.is_server_ready():
                raise Exception("Server is not ready")
            logger.info("Server is ready")
            
            # Проверяем доступность модели
            if not self.client.is_model_ready("text_normalization"):
                raise Exception("Model is not ready")
            logger.info("Model is ready")
            
        except Exception as e:
            logger.error(f"Error initializing Triton client: {str(e)}")
            raise

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

        try:
            preprocessed_text = self.preprocessor.preprocess_sentence(text)
            logger.info(f"Preprocessed text: {preprocessed_text}")

            inputs = self.tokenizer(
                preprocessed_text,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=self.max_input_length
            )
            logger.info("Successfully tokenized input")

            input_ids = httpclient.InferInput("input_ids", inputs["input_ids"].shape, "INT64")
            attention_mask = httpclient.InferInput("attention_mask", inputs["attention_mask"].shape, "INT64")

            input_ids.set_data_from_numpy(inputs["input_ids"].astype(np.int64))
            attention_mask.set_data_from_numpy(inputs["attention_mask"].astype(np.int64))
            logger.info("Successfully prepared input tensors")

            # Добавляем задержку перед запросом
            time.sleep(1)
            
            response = self.client.infer(
                model_name="text_normalization",
                inputs=[input_ids, attention_mask],
                outputs=[httpclient.InferRequestedOutput("output_ids")]
            )
            logger.info("Successfully received response from server")

            output_ids = response.as_numpy("output_ids")
            output_text = self.tokenizer.decode(output_ids[0][1:], skip_special_tokens=True)
            logger.info(f"Decoded output text: {output_text}")

            normalized_text = self.postprocessor.postprocess_outputs(text, output_text)
            logger.info(f"Final normalized text: {normalized_text}")

            return normalized_text
            
        except Exception as e:
            logger.error(f"Error during text normalization: {str(e)}")
            raise


def main():
    """Main function for testing the client."""
    test_text = "Революция␞1905 года␞потерпела␞поражение␞."
    try:
        logger.info("Starting test request")
        client = TritonClient()
        result = client.normalize_text(test_text)
        print(f"Original text: {test_text}")
        print(f"Normalized text: {result}")
    except Exception as e:
        logger.error(f"Error during test request: {str(e)}")
        raise


if __name__ == "__main__":
    main()
