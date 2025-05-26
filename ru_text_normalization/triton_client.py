import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np
from typing import List, Optional, Tuple
from utils.logger import logger
from utils.preprocessing import TextPreprocessor
from utils.postprocessing import TextPostprocessor


class TritonClient:
    """
    Client for interacting with Triton Inference Server.
    
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
        logger.info(f"Triton client initialized with URL: {url}")

    @staticmethod
    def prepare_input_tensors(text: str) -> List[httpclient.InferInput]:
        """
        Prepare input tensors for model.
        
        Args:
            text (str): Input text
            
        Returns:
            List[httpclient.InferInput]: List of input tensors
        """
        logger.info(f"Preparing input tensors for text: {text}")
        try:
            # Prepare input data
            input_data = np.array([text.encode('utf-8')], dtype=np.object_)
            
            # Create input tensors
            inputs = []
            inputs.append(httpclient.InferInput("input", input_data.shape, np_to_triton_dtype(input_data.dtype)))
            inputs[0].set_data_from_numpy(input_data)
            
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
            output_data = results.as_numpy("output")
            normalized_text = output_data[0].decode('utf-8')
            
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
