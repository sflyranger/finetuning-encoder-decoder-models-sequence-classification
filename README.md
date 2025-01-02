# Finetuning Encoder and Decoder Models for Emotion Detection

## Overview
This repository contains implementations and experiments for multilabel emotion detection using various pretrained language models. The task is to classify texts into multiple emotion categories using models of varying sizes and capabilities. Each model is evaluated based on its performance on the validation set and in competition settings.

Kaggle Competition Link: https://www.kaggle.com/competitions/emotion-detection-fall-2024

### Models Trained
The models are listed in the order of their training and evaluation:
1. **RoBERTa-base (encoder)**: A robust encoder-only model optimized for text classification tasks.
2. **DistilBERT-base-uncased (encoder)**: A distilled version of BERT, providing a lightweight yet efficient encoder for classification.
3. **DistilRoBERTa-base (encoder)**: A distilled version of RoBERTa, retaining the benefits of the RoBERTa architecture with improved training efficiency.
4. **Gemma (decoder with LoRA)**: A decoder model fine-tuned using LoRA (Low-Rank Adaptation) to reduce the number of trainable parameters and optimize for generative capabilities.
5. **Llama-3.2-1B (decoder with LoRA)**: A generative decoder model optimized for text generation tasks, fine-tuned using LoRA for efficient parameter adaptation.
6. **Stella_en_1.5B_v5 (decoder with LoRA)**: Another powerful decoder model using LoRA for fine-tuning, focusing on optimizing performance while reducing overfitting.
7. **Qwen 2.5_1.5B Base (decoder)**: A decoder model trained to balance performance on frequent and sparse emotion labels, with specific focus on improving generalization and handling of underrepresented classes.
8. **Llama-3.2-3B-it (decoder)**: A decoder model leveraging iterative fine-tuning and enhanced cross-validation techniques to achieve balanced performance across frequent and minority emotion classes.
9. **Qwen-14B-Instruct (decoder using LoRA on Runpod platform)**: An instruction-tuned decoder model designed to understand nuanced inter-label relationships, achieving a strong balance of precision and recall across frequent and sparse emotion classes.



## Performance Comparison

Below is a summary of the overall performance of each model based on the **F1-Macro** scores. This metric was chosen as the key evaluation criterion to assess model effectiveness across all classes, especially the underrepresented ones.

<img width="1190" alt="image" src="https://github.com/user-attachments/assets/41f4d042-ebd1-4619-809b-3a2eb2ecfba8" />


### Key Observations:
- The **Gemma model** showed the highest F1-Macro score (0.6222), particularly excelling in the minority classes like "anticipation" and "trust".
- The **Llama-3.2-1B** and **Stella models** performed comparably, with F1-Macro scores around 0.62, showcasing strengths in challenging classes like "anticipation" and "pessimism".
- **DistilRoBERTa-base** demonstrated a moderate improvement over its predecessors, especially for "fear" and "surprise."
- The initial **RoBERTa-base** model struggled due to challenges in class weight implementation, leading to poor performance on underrepresented classes.

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

### 4. Gemma
- **Notebook**: [`Gemma_2b_Multilabel_Classification.ipynb`](./Gemma_2b_Multilabel_Classification.ipynb)
- **Training Method**: Fine-tuned using the `transformers` library with custom class weight adjustments.
- **Validation Strategy**: 10-fold cross-validation to ensure robust evaluation.

- **Notes**:
  - Successfully implemented class weights by calculating them relative to their distribution within each class rather than spanning across all classes. This adjustment significantly improved the F1 scores, particularly for the underrepresented classes.
  - Noteworthy improvements were observed in the F1 scores for the minority classes like **"anticipation"**, **"surprise"**, and **"trust"**, which were previously challenging to predict accurately.

#### Results
- **Average F1-Macro**: 0.6222

- **Class-wise F1 Scores**:

| Label         | F1-Score |
|---------------|----------|
| anger         | 0.8409   |
| anticipation  | 0.4326   |
| disgust       | 0.7786   |
| fear          | 0.7681   |
| joy           | 0.8197   |
| love          | 0.6289   |
| optimism      | 0.7561   |
| pessimism     | 0.4417   |
| sadness       | 0.6937   |
| surprise      | 0.4      |
| trust         | 0.2839   |

- **Key Observations**:
  - The **F1 score for "trust"** improved to 0.2839, which is a significant jump compared to earlier models that struggled with this class.
  - The model achieved an **F1 score of 0.4326 for "anticipation"** and **0.4 for "surprise"**, marking substantial gains for these previously challenging classes.

### 5. Llama-3.2-1B
- **Notebook**: [`Llama_3_2_1B_emotion_detection.ipynb`](./Llama_3_2_1B_emotion_detection.ipynb)
- **Training Method**: Focused on optimizing the model with weight decay and extended training epochs to improve performance on minority classes.
- **Validation Strategy**: 10-fold cross-validation to ensure robust evaluation.

- **Notes**:
  - Continued refinement of class weights, focusing on better handling of underrepresented classes.
  - Significant improvements observed in predicting the classes "trust" and "surprise," which were historically difficult to classify.

#### Results
- **Average F1-Macro**: 0.6217

- **Class-wise F1 Scores**:

| Label         | F1-Score |
|---------------|----------|
| anger         | 0.8319   |
| anticipation  | 0.4351   |
| disgust       | 0.7754   |
| fear          | 0.75     |
| joy           | 0.8231   |
| love          | 0.6207   |
| optimism      | 0.7495   |
| pessimism     | 0.4607   |
| sadness       | 0.6897   |
| surprise      | 0.4545   |
| trust         | 0.2481   |

- **Key Observations**:
  - The **F1 score for "trust"** improved to 0.2481, a notable gain compared to previous models.
  - Achieved an **F1 score of 0.4545 for "surprise"** and **0.4351 for "anticipation"**, demonstrating progress in classifying these challenging categories.

### 6. Stella_en_1.5B_v5
- **Notebook**: [`stella_en_1_5B_v5_emotion_detection_multilabel_10.ipynb`](./stella_en_1_5B_v5_emotion_detection_multilabel_10.ipynb)
- **Training Method**: Utilized extended training epochs with weight decay and a custom `stella` tokenizer to optimize performance on underrepresented classes.
- **Validation Strategy**: 10-fold cross-validation to achieve robust evaluation.

- **Notes**:
  - Initially, the model trained for too long, leading to overfitting. To address this, I reran the evaluation using a checkpoint that I manually selected based on a combination of validation loss and F1 score.
  - The checkpoint I chose had a validation loss significantly lower (by 100%) than the checkpoint selected automatically during training, while the F1 score was only 0.002 lower than that of the default best checkpoint. This balance helped reduce overfitting while maintaining competitive F1 performance.
  - The model showed notable improvements in F1 scores for minority classes, especially "trust" and "anticipation," which were previously difficult to classify.

#### Results
- **Average F1-Macro**: 0.6191

- **Class-wise F1 Scores**:

| Label         | F1-Score |
|---------------|----------|
| anger         | 0.8154   |
| anticipation  | 0.4634   |
| disgust       | 0.75     |
| fear          | 0.7394   |
| joy           | 0.81     |
| love          | 0.6542   |
| optimism      | 0.7490   |
| pessimism     | 0.4053   |
| sadness       | 0.6866   |
| surprise      | 0.4      |
| trust         | 0.3371   |

- **Key Observations**:
  - By using a custom-selected checkpoint, the **F1 score for "trust"** improved to 0.3371, a significant gain for this challenging class.
  - The **F1 score for "anticipation"** also increased to 0.4634, marking progress in handling classes with sparse data.
  - The chosen checkpoint effectively balanced between minimizing validation loss and maximizing F1 score, reducing overfitting while preserving model performance.

### 7. Qwen 2.5_1.5B Base
- **Notebook**: [`Qwen_2.5_1.5B_Base.ipynb`](./Qwen_2_5_1_5B_multilabel_emotion_generation.ipynb)
- **Training Method**: Fine-tuned with a focus on balancing class weights and improving prediction for underrepresented classes.
- **Validation Strategy**: 10-fold cross-validation for robust performance assessment.

- **Notes**:
  - This model attempted to handle sparse data in minority classes through careful adjustment of loss weighting.
  - The validation results showed good precision for some classes but struggled with recall for several underrepresented emotions, particularly "disgust" and "trust."
  - Competition results reflect the model's challenges in generalizing to unseen data, with a lower overall F1 score.

#### Results
- **Average F1-Macro (Validation)**: 0.4341
- **Average F1-Macro (Competition)**: 0.4257

- **Class-wise F1 Scores (Validation)**:

| Label         | F1-Score |
|---------------|----------|
| anger         | 0.8261   |
| anticipation  | 0.4413   |
| disgust       | 0.0269   |
| fear          | 0.3799   |
| joy           | 0.8064   |
| love          | 0.6      |
| optimism      | 0.6387   |
| pessimism     | 0.2703   |
| sadness       | 0.4927   |
| surprise      | 0.1702   |
| trust         | 0.1224   |

- **Key Observations**:
  - Precision was high for most classes, but recall was very low for emotions like "disgust" and "trust," resulting in reduced F1 scores for those classes.
  - Despite struggles with underrepresented classes, the model demonstrated strong performance on "anger," "joy," and "optimism."
  - Competition performance revealed challenges with generalization, reflected in a slightly lower F1-macro score compared to validation.


 ### 8. Llama-3.2-3B-it
- **Notebook**: [`llama_3_2_3B_it_Multilabel__emotion_generation.ipynb`](./llama_3_2_3B_it_Multilabel__emotion_generation.ipynb)
- **Training Method**: Fine-tuned using the `transformers` library with enhanced training strategies to optimize performance for emotion classification.
- **Validation Strategy**: 10-fold cross-validation to ensure robust and balanced evaluation across all classes.

- **Notes**:
  - This model introduced iterative fine-tuning techniques, leveraging feedback from prior experiments to optimize hyperparameters and loss weighting.
  - Demonstrated a more balanced performance across classes compared to earlier models, particularly improving recall for minority emotions such as "disgust" and "trust."
  - Significant gains in precision and recall for "joy," "optimism," and "anger" highlight the model's ability to generalize well across both frequent and sparse labels.

#### Results
- **Average F1-Macro (Validation)**: 0.5929
- **Average F1-Macro (Competition)**: 0.5841

- **Class-wise F1 Scores (Validation)**:

| Label         | F1-Score |
|---------------|----------|
| anger         | 0.8169   |
| anticipation  | 0.4288   |
| disgust       | 0.7762   |
| fear          | 0.7491   |
| joy           | 0.8283   |
| love          | 0.5778   |
| optimism      | 0.7031   |
| pessimism     | 0.3547   |
| sadness       | 0.6939   |
| surprise      | 0.3474   |
| trust         | 0.2460   |

- **Key Observations**:
  - The **F1 score for "disgust"** reached 0.7762, marking one of the highest performances for this class among all models.
  - The model achieved an **F1 score of 0.4288 for "anticipation"** and **0.3474 for "surprise,"** indicating steady progress for these challenging classes.
  - Notable improvements in recall for "joy" and "optimism" demonstrate the model's robustness in handling positive emotions.
  - The Kaggle competition F1-Macro score of **0.5841** closely aligns with the validation results, showcasing the model's ability to generalize effectively to unseen data.


### 9. Qwen-14B-Instruct
- **Notebook**: [`QWEN_14B_Instruct_multilabel_generation_emotion.ipynb`](./QWEN_14B_Instruct_multilabel_generation_emotion.ipynb)
- **Training Method**: Fine-tuned using the `transformers` library with an emphasis on handling complex, multilabel emotion classification tasks. The model incorporated instruction-tuned methodologies to enhance understanding of nuanced label relationships.
- **Validation Strategy**: 10-fold cross-validation for robust and fair evaluation across all classes.

- **Notes**:
  - This model exhibited strong performance across frequent and sparse emotion labels, showing particular improvements in "disgust," "optimism," and "joy."
  - Leveraged instruction tuning to better handle interdependent label relationships, enabling balanced precision and recall across emotion classes.
  - Notable gains in minority class predictions, with improved F1 scores for "disgust," "surprise," and "trust."

#### Results
- **Average F1-Macro (Validation)**: 0.6229
- **Average F1-Macro (Competition)**: 0.5963

- **Class-wise F1 Scores (Validation)**:

| Label         | F1-Score |
|---------------|----------|
| anger         | 0.8279   |
| anticipation  | 0.4465   |
| disgust       | 0.7545   |
| fear          | 0.7365   |
| joy           | 0.8268   |
| love          | 0.6237   |
| optimism      | 0.7813   |
| pessimism     | 0.4590   |
| sadness       | 0.7070   |
| surprise      | 0.5373   |
| trust         | 0.1509   |

- **Key Observations**:
  - Achieved the **highest F1 score for "optimism"** (0.7813), marking a significant improvement compared to previous models.
  - The **F1 score for "disgust"** improved to 0.7545, demonstrating better handling of this challenging class.
  - Instruction tuning facilitated balanced performance across classes, with steady progress for both frequent and sparse labels like "surprise" and "trust."
  - The Kaggle competition F1-Macro score of **0.5963** further highlights the model's ability to generalize effectively to unseen data.


