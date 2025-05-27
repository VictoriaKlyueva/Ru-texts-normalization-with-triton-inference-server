import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np
from typing import List, Optional, Tuple
from transformers import GPT2Tokenizer
from ru_text_normalization.utils.logger import logger
from ru_text_normalization.utils.preprocessing import TextPreprocessor
from ru_text_normalization.utils.postprocessing import TextPostprocessor


class TritonClient:
    """
    Client for interacting with Triton Inference Server.
    
    Attributes:
        url (str): Triton server URL
        client (httpclient.InferenceServerClient): HTTP client for Triton
        preprocessor (TextPreprocessor): Text preprocessor
        postprocessor (TextPostprocessor): Text postprocessor
        tokenizer (GPT2Tokenizer): Tokenizer for text processing
    """
    
    def __init__(self, url: str = "localhost:8000", model_name: str = "vikosik3000/FRED_text_normalization"):
        """
        Initialize Triton client.
        
        Args:
            url (str): Triton server URL
            model_name (str): Name of the model to use for tokenization
        """
        self.url = url
        self.client = httpclient.InferenceServerClient(url=url)
        self.preprocessor = TextPreprocessor()
        self.postprocessor = TextPostprocessor()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        logger.info(f"Triton client initialized with URL: {url}")

    def prepare_input_tensors(self, text: str) -> List[httpclient.InferInput]:
        """
        Prepare input tensors for model.
        
        Args:
            text (str): Input text
            
        Returns:
            List[httpclient.InferInput]: List of input tensors
        """
        logger.info(f"Preparing input tensors for text: {text}")
        try:
            # Preprocess text
            preprocessed_text = self.preprocessor.preprocess_sentence(text)
            
            # Tokenize text
            input_ids = np.array([self.tokenizer.encode(preprocessed_text)], dtype=np.int64)
            attention_mask = np.ones_like(input_ids, dtype=np.int64)
            
            # Create input tensors
            inputs = []
            inputs.append(httpclient.InferInput("input_ids", input_ids.shape, np_to_triton_dtype(input_ids.dtype)))
            inputs[0].set_data_from_numpy(input_ids)
            
            inputs.append(httpclient.InferInput("attention_mask", attention_mask.shape, np_to_triton_dtype(attention_mask.dtype)))
            inputs[1].set_data_from_numpy(attention_mask)
            
            logger.info("Input tensors prepared successfully")
            return inputs
        except Exception as e:
            logger.error(f"Error preparing input tensors: {str(e)}")
            raise

    def normalize_text(self, text: str) -> str:
        """
        Normalize text using Triton Inference Server.
        
        Args:
            text (str): Input text for normalization
            
        Returns:
            str: Normalized text
        """
        logger.info(f"Starting text normalization: {text}")
        try:
            # Prepare input tensors
            inputs = self.prepare_input_tensors(text)
            
            # Send request
            logger.info("Sending request to server")
            results = self.client.infer("text_normalization", inputs)
            logger.info("Received response from server")

            # Get and process results
            logits = results.as_numpy("logits")
            output_ids = np.argmax(logits, axis=-1)
            output_text = self.tokenizer.decode(output_ids[0])
            
            # Postprocess output
            normalized_text = self.postprocessor.postprocess_outputs(text, output_text)
            
            logger.info(f"Normalization completed successfully. Result: {normalized_text}")
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
