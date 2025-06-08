# Ru text normalization
This project is a solution to a [Russian Text Normalization Challenge](https://www.kaggle.com/competitions/text-normalization-challenge-russian-language) Kaggle competition and Triton inference server solution based on it.

## The Data Science Part

Notebook: [notebook](https://colab.research.google.com/drive/1F5uJ9V5_pY8qXs-7z9qE9kmsubbrhCH1#scrollTo=HJsbTPf1rTkf)\
All project files on Google Drive: [project](https://drive.google.com/drive/folders/1e44PCViPQSdO-VEQimxptiJYGkPGz5eN?usp=sharing)\
Best score on Kaggle: **0.97591** on Public and **0.97519** on Private

<img src="https://github.com/VictoriaKlyueva/Ru-texts-normalization-triton-server/blob/readme/images/kaggle_leaderboard_screen">

### Screenshots from Wandb
<table>
<tbody>
  <tr>
    <td>Train metrics</td>
    <td>Validations metrics</td>
  </tr>
  <tr>
    <td><img src="https://github.com/VictoriaKlyueva/Ru-texts-normalization-triton-server/blob/readme/images/wandb_train.png"></td>
    <td><img src="https://github.com/VictoriaKlyueva/Ru-texts-normalization-triton-server/blob/readme/images/wandb_test.png"></td>
  </tr>
</tbody>
</table>

### Models metrics
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Private score</th>
      <th>Public score</th>
      <th>Comment</th>
      <th>Conclusion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Only rules</td>
      <td><strong>0.97591</strong></td>
      <td><strong>0.97519</strong></td>
      <td>Most common rules were implemented</td>
      <td>Rule based aproach really good for this task</td>
    </tr>
    <tr>
      <td>
        Finetuned <a href="https://huggingface.co/saarus72/russian_text_normalizer">FRED based text normalization model</a> <em>without</em> postprocessing
      </td>
      <td>0.92600</td>
      <td>0.92630</td>
      <td>Only simple postprocessing was implemented to parse results with right format</td>
      <td>Model can predict right tokens in most common non-typical cases using sentence context, but is has problems with presentation result in right format (for example, translation with "_" before all symbol) and with unambiguous but rare cases (for example, "ТГУ" -> "т г у")</td>
    </tr>
    <tr>
      <td>
        Finetuned <a href="https://huggingface.co/saarus72/russian_text_normalizer">FRED based text normalization model</a> <em>with</em> postprocessing
      </td>
      <td>0.96479</td>
      <td>0.96381</td>
      <td>Were added not all from "Only rules" aproach but most common rules (most of them for right presentation model answers for submits requirements) </td>
      <td>Using more rules may significantly increase model's answers quality</td>
    </tr>
    <tr>
      <td>Finetuned <a href="https://huggingface.co/ai-forever/ruT5-base">ruT5</a></td>
      <td>-</td>
      <td>-</td>
      <td>I trained model during about 3 hours on one A100 GPU, but model was whatever underfitted</td>
      <td>It's too resource-demanding and not effective to train model from scratch</td>
    </tr>
    <tr>
      <td>Few-shoted <a href="https://huggingface.co/IlyaGusev/saiga_llama3_8b">saiga_llama3_8b</a></td>
      <td>-</td>
      <td>-</td>
      <td>I tried to inference model with 18 different examples in system prompt, but model is too large to make submission on test in acceptable time and it is hard to postprocess model's answers</td>
      <td>LLMs are too complex for this task</td>
    </tr>
    <tr>
      <td>RoBERTa</a></td>
      <td>-</td>
      <td>-</td>
      <td>I tried solve problem as TokenClassification task, but i got problems with implemetation and just quit</td>
      <td>I think it was a bad idea initially :)</td>
    </tr>
  </tbody>
</table>

<TODO>

## Software Engineer part

It contains code for inference the model using Triton Inference Server

### How to run

1. Clone repository
  ```bash
  git clone https://github.com/VictoriaKlyueva/Ru-texts-normalization-triton-server.git
  ```

2. Run server
  ```bash
  docker-compose up --build
  ```

3. Install triton SDK and run tritonserver SDK
  ```bash
  docker pull nvcr.io/nvidia/tritonserver:24.08-py3-sdk
  docker run --gpus all --rm -it --net host nvcr.io/nvidia/tritonserver:24.08-py3-sdk
  ```

4. Run Perf Analyzer for tensorrt_llm model
  ```bash
  perf_analyzer -m tensorrt_llm model
  ```

## Author
- `Klyueva Victoria, 972302`
