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

    # Подготовка входных данных
    input_data = np.array([text.encode('utf-8')], dtype=np.object_)
    
    # Создание входных тензоров
    inputs = []
    inputs.append(httpclient.InferInput("input", input_data.shape, np_to_triton_dtype(input_data.dtype)))
    inputs[0].set_data_from_numpy(input_data)

    # Отправка запроса
    results = client.infer("text_normalization", inputs)

    # Получение и обработка результатов
    output_data = results.as_numpy("output")
    normalized_text = output_data[0].decode('utf-8')

    return normalized_text


if __name__ == "__main__":
    # Тестовый запрос
    test_text = "Привет, как дела?"
    try:
        result = normalize_text(test_text)
        print(f"Исходный текст: {test_text}")
        print(f"Нормализованный текст: {result}")
    except Exception as e:
        print(f"Ошибка при отправке запроса: {e}")
