# Multilabel Emotion Detection

## Overview
This repository contains implementations and experiments for multilabel emotion detection using various pretrained language models. The task is to classify texts into multiple emotion categories using models of varying sizes and capabilities. Each model is evaluated based on its performance on the validation set and in competition settings.

### Models Trained
The models are listed in the order of their training and evaluation:
1. `RoBERTa-base`
2. `DistilBERT-base-uncased`
3. `DistilRoBERTa-base`
4. `Gemma`
5. `Llama-3.2-1B`
6. `Stella_en_1.5B_v5`

## Model Training Details

### 1. RoBERTa-base
- **Notebook**: [`RoBERTA_BASE_EMOTION_DETECTION.ipynb`](./RoBERTA_BASE_EMOTION_DETECTION.ipynb)
- **Training Method**: Standard training with multilabel binary-cross-entropy loss using `transformers` and `torch` libraries.
- **Challenges**: Experienced issues with class weight implementation, leading to poor model performance, especially on minority classes.
- **Validation Strategy**: 5-fold MultilabelStratifiedKFold with macro-F1 scoring.

#### Results
- **Average F1-Macro**: 0.0479
- **Class-wise F1-Micro Scores**:

| Label         | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| anger         | 0.358    | 1.0    | 0.5272   |
| anticipation  | 0.0      | 0.0    | 0.0      |
| disgust       | 0.0      | 0.0    | 0.0      |
| fear          | 0.0      | 0.0    | 0.0      |
| joy           | 0.0      | 0.0    | 0.0      |
| love          | 0.0      | 0.0    | 0.0      |
| optimism      | 0.0      | 0.0    | 0.0      |
| pessimism     | 0.0      | 0.0    | 0.0      |
| sadness       | 0.0      | 0.0    | 0.0      |
| surprise      | 0.0      | 0.0    | 0.0      |
| trust         | 0.0      | 0.0    | 0.0      |

### 2. DistilBERT-base-uncased
- **Notebook**: [`distilBERT_BASE_UNCASED_ED.ipynb`](./distilBERT_BASE_UNCASED_ED.ipynb)
- **Training Method**: Similar to RoBERTa-base, with modifications for a smaller model size. Focused on improving efficiency while maintaining performance.
- **Validation Strategy**: 5-fold cross-validation using F1-macro as the primary metric.
  
- **Notes**:
  - This was the first model where I experimented with `MultilabelStratifiedKFold` to ensure better distribution of labels across folds.
  - Faced difficulties implementing class weights, which impacted performance, particularly for the minority and more disparate classes.
  - The model struggled to predict emotions like "anticipation," "surprise," and "trust," which had sparse occurrences in the dataset.

#### Results
- **Average F1-Macro**: 0.4658
- **Average Precision**: 0.5184
- **Average Recall**: 0.4358

- **Class-wise F1-Micro Scores**:

| Label         | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| anger         | 0.7671   | 0.7657 | 0.7664   |
| anticipation  | 0.4396   | 0.1818 | 0.2572   |
| disgust       | 0.7077   | 0.7077 | 0.7077   |
| fear          | 0.7159   | 0.7132 | 0.7145   |
| joy           | 0.7941   | 0.6962 | 0.7419   |
| love          | 0.5794   | 0.4371 | 0.4983   |
| optimism      | 0.6711   | 0.5568 | 0.6086   |
| pessimism     | 0.44     | 0.1844 | 0.2598   |
| sadness       | 0.5882   | 0.5507 | 0.5688   |
| surprise      | 0.0      | 0.0    | 0.0      |
| trust         | 0.0      | 0.0    | 0.0      |

### 3. DistilRoBERTa-base
- **Notebook**: [`distilRoBERTa_base_emotion_detection.ipynb`](./distilRoBERTa_base_emotion_detection.ipynb)
- **Training Method**: This model was optimized as a distilled version of RoBERTa, focusing on efficiency and faster training times.
- **Validation Strategy**: 
  - Implemented `MultilabelStratifiedKFold` with 10-fold splits, which significantly improved model stability and performance.
  - Noticed that the model was generalizing well even after the initial training phase of 3 epochs. Therefore, extended the training to further optimize performance.
  
- **Notes**:
  - The use of 10-fold cross-validation allowed for a more even distribution of labels across folds, leading to better generalization.
  - The model showed noticeable improvements in predicting less frequent classes such as "surprise" and "anticipation," though still struggled with "trust."

#### Results
- **Average F1-Macro**: 0.5587
- **Average Precision**: 0.6362
- **Average Recall**: 0.5232

- **Class-wise F1-Micro Scores**:

| Label         | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| anger         | 0.8191   | 0.8105 | 0.8148   |
| anticipation  | 0.5556   | 0.3153 | 0.4023   |
| disgust       | 0.7582   | 0.7089 | 0.7327   |
| fear          | 0.7248   | 0.7883 | 0.7552   |
| joy           | 0.8476   | 0.7917 | 0.8187   |
| love          | 0.6286   | 0.5301 | 0.5752   |
| optimism      | 0.7200   | 0.7860 | 0.7516   |
| pessimism     | 0.4909   | 0.3    | 0.3724   |
| sadness       | 0.7256   | 0.5242 | 0.6087   |
| surprise      | 0.7273   | 0.2    | 0.3137   |
| trust         | 0.0      | 0.0    | 0.0      |

