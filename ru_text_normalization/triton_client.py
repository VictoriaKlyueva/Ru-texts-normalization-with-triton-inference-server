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
    Client for Triton Inference Server.
    
    Attributes:
        client (httpclient.InferenceServerClient): Triton client
        preprocessor (TextPreprocessor): Text preprocessor
        postprocessor (TextPostprocessor): Text postprocessor
        tokenizer (GPT2Tokenizer): Tokenizer for text processing
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
        self.tokenizer = GPT2Tokenizer.from_pretrained('saarus72/russian_text_normalizer', eos_token='</s>')
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
            # Prepare inputs for tensorrt_llm model
            inputs = [
                httpclient.InferInput("input_ids", [-1], "INT32"),
                httpclient.InferInput("input_lengths", [1], "INT32"),
                httpclient.InferInput("request_output_len", [1], "INT32"),
                httpclient.InferInput("exclude_input_in_output", [1], "BOOL"),
            ]

            # Tokenize input text
            encoded = self.tokenizer(
                input_text,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=self.max_input_length
            )
            input_ids = encoded["input_ids"][0]
            input_ids = input_ids.reshape(-1).astype(np.int32)
            
            # Set input values
            inputs[0].set_data_from_numpy(input_ids)
            inputs[1].set_data_from_numpy(np.array([len(input_ids)], dtype=np.int32))
            inputs[2].set_data_from_numpy(np.array([self.max_output_length], dtype=np.int32))
            inputs[3].set_data_from_numpy(np.array([True], dtype=np.bool_))  # exclude_input_in_output

            # Prepare outputs
            outputs = [
                httpclient.InferRequestedOutput("output_ids"),
                httpclient.InferRequestedOutput("sequence_length"),
            ]

            logger.info(f"Sending request to model with input: {input_text}")
            # Execute inference
            response = self.client.infer(
                "tensorrt_llm",
                inputs=inputs,
                outputs=outputs
            )

            # Process response
            output_ids = response.as_numpy("output_ids")
            sequence_length = response.as_numpy("sequence_length")
            
            # Decode output_ids to text
            output_text = self.tokenizer.decode(output_ids[0][:sequence_length[0]], skip_special_tokens=True)
            logger.info(f"Raw model output: {output_text}")

            return output_text

        except Exception as e:
            logger.error(f"Error during model inference: {str(e)}")
            raise

    def normalize_text(self, text: str) -> str:
        """
        Normalize text using the model.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Normalized text
        """
        try:
            # Preprocess input text
            input_text = self.preprocessor.preprocess_sentence(text)
            logger.info(f"Preprocessed text: {input_text}")
            
            # Get model output
            output = self.get_model_output(input_text)
            logger.info(f"Raw model output: {output}")
            
            # Postprocess output
            normalized_text = self.postprocessor.postprocess_outputs(text, output)
            logger.info(f"Postprocessed text: {normalized_text}")
            
            logger.info(f"Text normalization completed successfully")
            return normalized_text
            
        except Exception as e:
            logger.error(f"Error during text normalization: {str(e)}")
            raise


def main():
    """Main function for testing."""
    try:
        # Initialize client
        client = TritonClient()
        
        # Test text
        test_text = "Революция 1905 года потерпела поражение."
        logger.info(f"Input text: {test_text}")
        
        # Normalize text
        normalized_text = client.normalize_text(test_text)
        logger.info(f"Normalized text: {normalized_text}")
        
    except Exception as e:
        logger.error(f"Error during test request: {str(e)}")
        raise


if __name__ == "__main__":
    main()
