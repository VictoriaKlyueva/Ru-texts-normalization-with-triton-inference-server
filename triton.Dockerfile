FROM nvcr.io/nvidia/tritonserver:22.12-py3

WORKDIR /app

# Копирование файлов проекта
COPY pyproject.toml ./
COPY ru_text_normalization/model_repository /models

# Установка зависимостей
RUN pip install --no-cache-dir numpy>=1.21.0 pandas>=1.3.0

# Run Triton server
CMD ["tritonserver", "--model-repository=/models"]