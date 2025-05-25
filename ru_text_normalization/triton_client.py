import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np


def normalize_text(text, url="localhost:8000"):
    """
    Normalizes text using a model deployed on Triton Inference Server.
    
    Args:
        text (str): Input text for normalization
        url (str): Triton server URL
        
    Returns:
        str: Normalized text
    """
    client = httpclient.InferenceServerClient(url=url)
    
    # Text preprocessing (adapt to your model)
    input_data = preprocess_text(text)
    
    # Prepare input data
    inputs = []
    inputs.append(httpclient.InferInput("input_ids", input_data.shape, np_to_triton_dtype(input_data.dtype)))
    inputs[0].set_data_from_numpy(input_data)
    
    # Send request
    results = client.infer("text_normalization", inputs)
    
    # Get and post-process results
    output_data = results.as_numpy("output")
    normalized_text = postprocess_output(output_data)
    
    return normalized_text
