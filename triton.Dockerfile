FROM nvcr.io/nvidia/tritonserver:22.12-py3

WORKDIR /app

# Копирование файлов проекта
COPY requirements.txt ./
COPY ru_text_normalization/model_repository /models

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Run Triton server
CMD ["tritonserver", "--model-repository=/models"]