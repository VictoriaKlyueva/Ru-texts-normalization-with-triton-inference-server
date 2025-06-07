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
            hf_model_name (str): Name of the model to use for tokenization
        """
        self.url = url
        self.client = httpclient.InferenceServerClient(url=url)
        self.preprocessor = TextPreprocessor()
        self.postprocessor = TextPostprocessor()
        self.max_input_length = 256
        self.max_output_length = 256

        logger.info(f"Triton client initialized with URL: {url}")

    def normalize_text(
        self,
        text: str,
        max_tokens: int = 256,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        stop_words: Optional[List[str]] = None,
        bad_words: Optional[List[str]] = None,
        exclude_input_in_output: bool = True,
        return_log_probs: bool = False,
        return_context_logits: bool = False,
        return_generation_logits: bool = False,
        return_perf_metrics: bool = False,
        stream: bool = False,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Normalize text using Triton Inference Server with T5 model.

        Args:
            text (str): Input text for normalization
            max_tokens (int): Maximum number of tokens to generate
            num_return_sequences (int): Number of sequences to return
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling parameter
            top_p (float): Top-p sampling parameter
            repetition_penalty (float): Penalty for repetition
            length_penalty (float): Penalty for sequence length
            presence_penalty (float): Penalty for presence of tokens
            frequency_penalty (float): Penalty for token frequency
            stop_words (List[str]): List of stop words
            bad_words (List[str]): List of bad words
            exclude_input_in_output (bool): Whether to exclude input in output
            return_log_probs (bool): Whether to return log probabilities
            return_context_logits (bool): Whether to return context logits
            return_generation_logits (bool): Whether to return generation logits
            return_perf_metrics (bool): Whether to return performance metrics
            stream (bool): Whether to stream the response
            seed (int): Random seed for generation

        Returns:
            Dict[str, Any]: Dictionary containing normalized text and additional outputs
        """
        logger.info(f"Starting text normalization for text: {text}")
        try:
            # Prepare inputs for ensemble model
            inputs = [
                httpclient.InferInput("text_input", [1, 1], "BYTES"),
                httpclient.InferInput("max_tokens", [1, 1], "INT32"),
                httpclient.InferInput("num_return_sequences", [1, 1], "INT32"),
                httpclient.InferInput("temperature", [1, 1], "FP32"),
                httpclient.InferInput("top_k", [1, 1], "INT32"),
                httpclient.InferInput("top_p", [1, 1], "FP32"),
                httpclient.InferInput("repetition_penalty", [1, 1], "FP32"),
                httpclient.InferInput("length_penalty", [1, 1], "FP32"),
                httpclient.InferInput("presence_penalty", [1, 1], "FP32"),
                httpclient.InferInput("frequency_penalty", [1, 1], "FP32"),
                httpclient.InferInput("exclude_input_in_output", [1, 1], "BOOL"),
                httpclient.InferInput("return_log_probs", [1, 1], "BOOL"),
                httpclient.InferInput("return_context_logits", [1, 1], "BOOL"),
                httpclient.InferInput("return_generation_logits", [1, 1], "BOOL"),
                httpclient.InferInput("return_perf_metrics", [1, 1], "BOOL"),
                httpclient.InferInput("stream", [1, 1], "BOOL"),
            ]

            # Set input values
            inputs[0].set_data_from_numpy(np.array([[text.encode()]], dtype=object))
            inputs[1].set_data_from_numpy(np.array([[max_tokens]], dtype=np.int32))
            inputs[2].set_data_from_numpy(np.array([[num_return_sequences]], dtype=np.int32))
            inputs[3].set_data_from_numpy(np.array([[temperature]], dtype=np.float32))
            inputs[4].set_data_from_numpy(np.array([[top_k]], dtype=np.int32))
            inputs[5].set_data_from_numpy(np.array([[top_p]], dtype=np.float32))
            inputs[6].set_data_from_numpy(np.array([[repetition_penalty]], dtype=np.float32))
            inputs[7].set_data_from_numpy(np.array([[length_penalty]], dtype=np.float32))
            inputs[8].set_data_from_numpy(np.array([[presence_penalty]], dtype=np.float32))
            inputs[9].set_data_from_numpy(np.array([[frequency_penalty]], dtype=np.float32))
            inputs[10].set_data_from_numpy(np.array([[exclude_input_in_output]], dtype=np.bool_))
            inputs[11].set_data_from_numpy(np.array([[return_log_probs]], dtype=np.bool_))
            inputs[12].set_data_from_numpy(np.array([[return_context_logits]], dtype=np.bool_))
            inputs[13].set_data_from_numpy(np.array([[return_generation_logits]], dtype=np.bool_))
            inputs[14].set_data_from_numpy(np.array([[return_perf_metrics]], dtype=np.bool_))
            inputs[15].set_data_from_numpy(np.array([[stream]], dtype=np.bool_))

            # Add optional inputs if provided
            if stop_words:
                inputs.append(httpclient.InferInput("stop_words", [1, len(stop_words)], "BYTES"))
                inputs[-1].set_data_from_numpy(np.array([[w.encode() for w in stop_words]], dtype=object))

            if bad_words:
                inputs.append(httpclient.InferInput("bad_words", [1, len(bad_words)], "BYTES"))
                inputs[-1].set_data_from_numpy(np.array([[w.encode() for w in bad_words]], dtype=object))

            if seed is not None:
                inputs.append(httpclient.InferInput("seed", [1, 1], "UINT64"))
                inputs[-1].set_data_from_numpy(np.array([[seed]], dtype=np.uint64))

            # Prepare outputs
            outputs = [
                httpclient.InferRequestedOutput("text_output"),
                httpclient.InferRequestedOutput("cum_log_probs"),
            ]

            if return_log_probs:
                outputs.append(httpclient.InferRequestedOutput("output_log_probs"))
            if return_context_logits:
                outputs.append(httpclient.InferRequestedOutput("context_logits"))
            if return_generation_logits:
                outputs.append(httpclient.InferRequestedOutput("generation_logits"))
            if return_perf_metrics:
                outputs.extend([
                    httpclient.InferRequestedOutput("arrival_time_ns"),
                    httpclient.InferRequestedOutput("first_scheduled_time_ns"),
                    httpclient.InferRequestedOutput("first_token_time_ns"),
                    httpclient.InferRequestedOutput("last_token_time_ns"),
                ])

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

            result = {
                "normalized_text": normalized_text,
                "cum_log_probs": response.as_numpy("cum_log_probs")[0][0] if return_log_probs else None,
            }

            if return_log_probs:
                result["output_log_probs"] = response.as_numpy("output_log_probs")
            if return_context_logits:
                result["context_logits"] = response.as_numpy("context_logits")
            if return_generation_logits:
                result["generation_logits"] = response.as_numpy("generation_logits")
            if return_perf_metrics:
                result["metrics"] = {
                    "arrival_time_ns": response.as_numpy("arrival_time_ns")[0][0],
                    "first_scheduled_time_ns": response.as_numpy("first_scheduled_time_ns")[0][0],
                    "first_token_time_ns": response.as_numpy("first_token_time_ns")[0][0],
                    "last_token_time_ns": response.as_numpy("last_token_time_ns")[0][0],
                }

            logger.info(f"Normalization completed successfully. Result: {result['normalized_text']}")
            return result

        except Exception as e:
            logger.error(f"Error during text normalization: {str(e)}")
            raise


def main():
    """Main function for testing the client."""
    test_text = "Революция␞1905 года␞потерпела␞поражение␞."
    try:
        logger.info("Starting test request")
        client = TritonClient()
        result = client.normalize_text(
            test_text,
            max_tokens=256,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            return_log_probs=True
        )
        print(f"Original text: {test_text}")
        print(f"Normalized text: {result['normalized_text']}")
        print(f"Log probabilities: {result['cum_log_probs']}")
    except Exception as e:
        logger.error(f"Error during test request: {str(e)}")
        raise


if __name__ == "__main__":
    main()
