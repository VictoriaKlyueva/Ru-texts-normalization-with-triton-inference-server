FROM nvcr.io/nvidia/tritonserver:24.08-trtllm-python-py3

WORKDIR /app

# Copy project files
COPY requirements.txt ./
COPY ru_text_normalization/model_repository /models
COPY ru_text_normalization/engines /engines
COPY ru_text_normalization/hf_model /hf_model

# Install requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Run Triton server
CMD ["tritonserver", "--model-repository=/models"]