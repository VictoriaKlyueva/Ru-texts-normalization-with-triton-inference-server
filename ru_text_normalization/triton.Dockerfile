FROM nvcr.io/nvidia/tritonserver:22.12-py3

WORKDIR /app
COPY model_repository /models

COPY requirements.txt .
RUN pip install -r requirements.txt

# Run Triton server
CMD ["tritonserver", "--model-repository=/models"]