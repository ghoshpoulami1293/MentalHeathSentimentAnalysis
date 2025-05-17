# MentalHeathSentimentAnalysis

This study investigates the use of sentiment analysis to identify mental health triggers in user-generated content. The basic purpose is to categorize text into mental health-related categories and identify probable stressors connected with each. Using advanced natural language processing (NLP) approaches such as fine-tuned transformer models and zero-shot classification, this research achieves considerable accuracy in mental health predictions. It also discusses crucial issues including imbalanced
datasets and the complexities of nuanced sentiment analysis. The data are provided as detailed visualizations, providing actionable insights to inform demographic mental health studies and tailored interventions.


# Code1_EmotionSentimentClassification.ipynb

## Features and Workflow

### 1. **Data Preprocessing**
    - Remove irrelevant or inappropriate rows (e.g., suicidal instances).
    - Preprocess text data using lemmatization and stopword removal

### 2. **Model Training**
    - Utilize roberta-large pre-trained transformer model.
    - Fine-tune the model using Hugging Face's Trainer API on the preprocessed dataset.

### 3. **Evaluation**
    - Calculate and display metrics (accuracy, precision, recall, F1 score).
    - Generate confusion matrix for understanding misclassifications.
    - Outputs: Metrics, visualizations, and predictions saved in traintest_EmotionClassification_predictions.csv


# Code2_PredictedEmotionLabelsUsingTestSet

### 1. **Visualization**
    - Generate a bar plot of predicted label distributions for intuitive understanding after generating label predictions on the test set MentalHealth_TestSet.csv.
    - Provide detailed results in a tabular format.


# Code3_CauseAnalysisAndUrgencyDetection.ipynb

This notebook focuses on utilizing pre-trained transformers to analyze text for cause detection and predict scenarios requiring assistance based on mental health data.
The approach leverages zero-shot classification, fine-tuning, and visualization.

## Features and Workflow

### 1. **Data Preparation**
        - Load and preprocess mental health datasets.
        - Remove unnecessary columns and tokenize text fields.
        - Ensures appropriate formatting for transformer models.

### 2. **Zero-Shot Classification**
        - Apply a zero-shot classification model(roberta-large-mnli) to assign initial categories to the text data.
        - Categories include causes such as "Health Issues," "Relationship Issues," and more.
        - Results are stored in a CSV file (zero_shot_predictions.csv) for downstream tasks.

### 3. **Fine-Tuning a Pre-Trained Transformer**
        - Fine-tune a transformer model (`mental-bert-base-uncased`) using the zero-shot classification results as labels.
        - The entire dataset is used for training without splitting.
        - Save the fine-tuned model in fine_tuned_mentalbert_cause_classifier for predictions.

### 4. **Prediction on New Data**
        - Uses the fine-tuned model to classify new text data into cause categories.
        - Adds predicted cause labels to a dataset and saves the results.

### 5. **Urgency Detection**
        - Flag scenarios based on urgency scores, with a threshold to recommend assistance for urgent cases.
        - Visualize the count of cases requiring assistance versus those that do not.

### 6. **Visualizations**
        - Create bar charts to visualize category distribution and urgency flags in the dataset.

## Files
    - zero_shot_predictions.csv: Zero-shot classification results.
    - fine_tuned_mentalbert_cause_classifier/: Directory containing the fine-tuned model.
    - test_data_with_predicted_cause.csv: File with predicted causes for the test dataset.
    - test_data_with_urgency_flags.csv: Final output file with urgency flags.
    - MentalHealth_TestSet.csv - Evaluation test set 
    - MentalHealthDataset.csv - Training Dataset 
    - traintest_EmotionClassification_predictions.csv - Dataset populated to check the predicted Label and Calculate metrics in Code1_EmotionSentimentClassification
    - TestDatasetPredictions.csv - Label Prediction using Evaluation test set in Code2_PredictedEmotionLabelsUsingTestSet

## Dependencies
    - Python 3.8+
    - Transformers
    - Pandas
    - Scikit-learn
    - Matplotlib
    - Datasets
    - SpaCy

## Order of execution 
        - Code1_EmotionSentimentClassification.ipynb - Fine tune roberta to determine Labels for the posts in the dataset 
        - Code2_PredictedEmotionLabelsUsingTestSet - generate Labels for posts in the evaluation test set
        - Code2.1_Prompting_EmotionClassification - generate Labels for posts using examples to demonstrate functionality
        - Code3_CauseAnalysisAndUrgencyDetection.ipynb - code to determine Cause of Stress and Assign Urgency Labels by zero shot classification and Fine tuning pretrained models
        - Code3.1_PromptZeroShotClassification_GenerateCause - code to determine Cause of Stress using zero_shot_learning using examples
        - Code3.2_PromptFineTunedRoberta_GeneratePredictedCause - code to determine Cause of Stress using fine_tuned_mentalbert_cause_classifier using examples
        - Code3.3_PromptingToGetScore_EntireFunctionality - code to determine Cause of Stress, and assign scores and Flags using examples

## Environment file created using the following details to run this code in Narnia: environment_nlpProject.yaml
