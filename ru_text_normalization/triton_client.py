import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np
from typing import Tuple
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
        tokenizer (AutoTokenizer): Tokenizer for text processing
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
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, eos_token='</s>')
        self.max_input_length = 256
        self.max_output_length = 256

        # Special tokens from config
        self.pad_token_id = self.tokenizer.pad_token_id  # 0
        self.eos_token_id = self.tokenizer.eos_token_id  # 2
        self.decoder_start_token_id = 0  # From config.json

        logger.info(f"Triton client initialized with URL: {url}")

    def prepare_encoder_inputs(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare encoder input tensors.

        Args:
            text (str): Input text

        Returns:
            Tuple[np.ndarray, np.ndarray]: input_ids and attention_mask
        """
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length
        )
        return inputs['input_ids'].astype(np.int64), inputs['attention_mask'].astype(np.int64)

    @staticmethod
    def prepare_decoder_inputs(decoder_input_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare decoder input tensors.

        Args:
            decoder_input_ids (np.ndarray): Decoder input ids

        Returns:
            Tuple[np.ndarray, np.ndarray]: decoder_input_ids and decoder_attention_mask
        """
        decoder_attention_mask = np.ones_like(decoder_input_ids, dtype=np.int64)
        return decoder_input_ids, decoder_attention_mask

    def generate_sequence(self, encoder_inputs: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Generate output sequence using iterative decoding.

        Args:
            encoder_inputs (Tuple[np.ndarray, np.ndarray]): encoder input_ids and attention_mask

        Returns:
            np.ndarray: Generated output sequence
        """
        input_ids, attention_mask = encoder_inputs

        # Initialize decoder with start token
        decoder_input_ids = np.array([[self.decoder_start_token_id]], dtype=np.int64)
        decoder_inputs = self.prepare_decoder_inputs(decoder_input_ids)

        generated_sequence = []

        for _ in range(self.max_output_length):
            # Prepare inference inputs
            inputs = [
                httpclient.InferInput("input_ids", input_ids.shape, np_to_triton_dtype(input_ids.dtype)),
                httpclient.InferInput("attention_mask", attention_mask.shape, np_to_triton_dtype(attention_mask.dtype)),
                httpclient.InferInput("decoder_input_ids", decoder_inputs[0].shape,
                                      np_to_triton_dtype(decoder_inputs[0].dtype)),
                httpclient.InferInput("decoder_attention_mask", decoder_inputs[1].shape,
                                      np_to_triton_dtype(decoder_inputs[1].dtype)),
            ]

            inputs[0].set_data_from_numpy(input_ids)
            inputs[1].set_data_from_numpy(attention_mask)
            inputs[2].set_data_from_numpy(decoder_inputs[0])
            inputs[3].set_data_from_numpy(decoder_inputs[1])

            # Execute inference
            response = self.client.infer(
                "text_normalization",
                inputs=inputs,
                outputs=[httpclient.InferRequestedOutput("logits")]
            )

            # Get next token
            logits = response.as_numpy("logits")
            next_token = np.argmax(logits[0, -1, :])

            # Check for EOS
            if next_token == self.eos_token_id:
                break

            generated_sequence.append(next_token)

            # Update decoder inputs for next step
            decoder_input_ids = np.array([[next_token]], dtype=np.int64)
            decoder_inputs = self.prepare_decoder_inputs(decoder_input_ids)

        return np.array(generated_sequence)

    def normalize_text(self, text: str) -> str:
        """
        Normalize text using Triton Inference Server with T5 model.

        Args:
            text (str): Input text for normalization

        Returns:
            str: Normalized text
        """
        logger.info(f"Starting text normalization for text: {text}")
        try:
            # Preprocess text
            preprocessed_text = self.preprocessor.preprocess_sentence(text)

            logger.info(f"Preprocessed text: {preprocessed_text}")

            # Prepare encoder inputs
            encoder_inputs = self.prepare_encoder_inputs(preprocessed_text)

            # Generate output sequence
            output_ids = self.generate_sequence(encoder_inputs)

            # Decode and postprocess
            output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            logger.info(f"Model output: {preprocessed_text}")

            normalized_text = self.postprocessor.postprocess_outputs(text, output_text)

            logger.info(f"Normalization completed successfully. Result: {normalized_text}")
            return normalized_text

        except Exception as e:
            logger.error(f"Error during text normalization: {str(e)}")
            raise


def main():
    """Main function for testing the client."""
    test_text1 = "Революция␞1905 года␞потерпела␞поражение␞."
    try:
        logger.info("Starting test request")
        client = TritonClient()
        result = client.normalize_text(test_text1)
        print(f"Original text: {test_text1}")
        print(f"Normalized text: {result}")
    except Exception as e:
        logger.error(f"Error during test request: {str(e)}")
        raise


if __name__ == "__main__":
    main()
