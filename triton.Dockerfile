FROM nvcr.io/nvidia/tritonserver:22.12-py3

WORKDIR /app

# Копирование файлов проекта
COPY requirements.txt ./

# Установка зависимостей
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu117

COPY ru_text_normalization/model_repository /models

# Проверка наличия GPU
RUN python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('CUDA device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# Run Triton server with verbose logging
CMD ["tritonserver", "--model-repository=/models", "--strict-model-config=false", "--log-verbose=1", "--log-info=1"]
