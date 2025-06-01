FROM nvcr.io/nvidia/tritonserver:22.12-py3

WORKDIR /app

# Копирование файлов проекта
COPY requirements.txt ./

# Установка зависимостей
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu117

COPY ru_text_normalization/model_repository /models

# Run Triton server
CMD ["tritonserver", "--model-repository=/models", "--strict-model-config=false"]
