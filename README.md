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

## Software Engineer part

It contains code for inference the model using Triton Inference Server

### Used tools
- Poetry
- ClearML
- Optuna
- Catboost
- Git
