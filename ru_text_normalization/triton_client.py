import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from transformers import GPT2Tokenizer
from ru_text_normalization.utils.logger import logger
from ru_text_normalization.utils.preprocessing import TextPreprocessor
from ru_text_normalization.utils.postprocessing import TextPostprocessor


class TritonClient:
    """
    Client for interacting with Triton Inference Server for T5 text normalization.

    Attributes:
        url (str): Triton server URL
        client (httpclient.InferenceServerClient): HTTP client for Triton
        preprocessor (TextPreprocessor): Text preprocessor
        postprocessor (TextPostprocessor): Text postprocessor
    """

    def __init__(self, url: str = "localhost:8000"):
        """
        Initialize Triton client.

        Args:
            url (str): Triton server URL
        """
        self.url = url
        self.client = httpclient.InferenceServerClient(url=url)
        self.preprocessor = TextPreprocessor()
        self.postprocessor = TextPostprocessor()
        self.max_input_length = 256
        self.max_output_length = 256

        logger.info(f"Triton client initialized with URL: {url}")

    def get_model_output(self, input_text: str) -> str:
        """
        Get model output for input text.

        Args:
            input_text (str): Input text

        Returns:
            str: Generated text
        """
        try:
            # Prepare inputs for ensemble model
            inputs = [
                httpclient.InferInput("text_input", [1, 1], "BYTES"),
                httpclient.InferInput("max_tokens", [1, 1], "INT32"),
                httpclient.InferInput("temperature", [1, 1], "FP32"),
                httpclient.InferInput("top_k", [1, 1], "INT32"),
                httpclient.InferInput("top_p", [1, 1], "FP32"),
                httpclient.InferInput("repetition_penalty", [1, 1], "FP32"),
                httpclient.InferInput("length_penalty", [1, 1], "FP32"),
                httpclient.InferInput("exclude_input_in_output", [1, 1], "BOOL"),
            ]

            # Set input values
            inputs[0].set_data_from_numpy(np.array([[input_text.encode()]], dtype=object))
            inputs[1].set_data_from_numpy(np.array([[256]], dtype=np.int32))  # max_tokens
            inputs[2].set_data_from_numpy(np.array([[1.0]], dtype=np.float32))  # temperature
            inputs[3].set_data_from_numpy(np.array([[50]], dtype=np.int32))  # top_k
            inputs[4].set_data_from_numpy(np.array([[0.9]], dtype=np.float32))  # top_p
            inputs[5].set_data_from_numpy(np.array([[1.0]], dtype=np.float32))  # repetition_penalty
            inputs[6].set_data_from_numpy(np.array([[1.0]], dtype=np.float32))  # length_penalty
            inputs[7].set_data_from_numpy(np.array([[True]], dtype=np.bool_))  # exclude_input_in_output

            # Prepare outputs
            outputs = [
                httpclient.InferRequestedOutput("text_output"),
            ]

            # Execute inference
            response = self.client.infer(
                "ensemble",
                inputs=inputs,
                outputs=outputs
            )

            # Process response
            text_output = response.as_numpy("text_output")
            if isinstance(text_output[0][0], bytes):
                normalized_text = text_output[0][0].decode()
            else:
                normalized_text = str(text_output[0][0])

            return normalized_text

        except Exception as e:
            logger.error(f"Error during model inference: {str(e)}")
            raise

    def normalize_text(self, text: str) -> str:
        """
        Complete text normalization.

        Args:
            text (str): Input text

        Returns:
            str: Normalized text
        """
        try:
            # Preprocess input text
            input_text = self.preprocessor.preprocess_sentence(text)
            
            # Get model output
            output = self.get_model_output(input_text)
            
            # Postprocess output
            normalized_text = self.postprocessor.postprocess_outputs(text, output)
            
            logger.info(f"Text normalization completed successfully")
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
        normalized_text = client.normalize_text(test_text)
        print(f"Original text: {test_text}")
        print(f"Normalized text: {normalized_text}")
    except Exception as e:
        logger.error(f"Error during test request: {str(e)}")
        raise


if __name__ == "__main__":
    main()
