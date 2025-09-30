## README: Text Origin Classification

### Project Overview

This project addresses the growing prevalence of machine-generated text in various domains. Large Language Models (LLMs) such as ChatGPT, Claude, and Gemini have made text generation and refinement highly accessible. The goal of this task is to train a robust classifier to distinguish between human-written and machine-generated text.

### Dataset

The dataset is provided as a collection of dataframes, each containing three key columns:

| Column | Description |
| :--- | :--- |
| `id` | A unique identifier for each text sample. |
| `text` | The text content to be classified. |
| `label` | The target variable. A value of **1** indicates the text is machine-generated, and **0** indicates it is human-written. (Present only in the training and validation sets.) |

We provide a large volume of data; however, practitioners are encouraged to judiciously select the amount of data necessary for effective model training.

### Task

The core task is to predict the `label` for the text samples in the provided test data file.

**Submission Requirements:**
The final predictions must be submitted as a `.csv` file with two columns:
1.  `id`: Must match the identifiers in the test data.
2.  `label`: The predicted class (0 or 1) for the corresponding text.

### Solution Pipeline

We developed a classification pipeline combining feature engineering with a Logistic Regression model.

#### 1. Feature Engineering and Extraction

We utilized a **TF-IDF Vectorizer** for feature extraction, operating at both the **word level** and the **character level**. This approach captures different granularities of linguistic patterns that may differentiate human from machine-generated text.

Features tuned within the vectorizer included:
* **`max_features`**: The maximum number of features to be extracted.
* **`ngram_range`**: The range of n-grams (e.g., unigrams, bigrams, trigrams) to consider.
* **`stop_words`**: Inclusion or exclusion of common stop words.
* **Word Frequency**: Thresholds for minimum and maximum document frequency.

#### 2. Modeling and Tuning

**Model:** Logistic Regression.

**Addressing Imbalance:**
The initial dataset exhibited class imbalance (unequal distribution of human vs. machine-generated text). To ensure robust parameter optimization, we created a **balanced sampled dataset** specifically for hyperparameter tuning.

**Hyperparameter Tuning:**
We employed **GridSearch** to systematically fine-tune parameters for both the feature extractor and the classifier.

* **TF-IDF Vectorizer Parameters:** (as listed above)
* **Logistic Regression Parameters:**
    * `solver`: Algorithm to use in the optimization problem.
    * `penalty`: The norm used for regularization (e.g., L1, L2).
    * `class_weight`: Adjusting weights to account for class imbalance.
    * Regularization Strength (`C`): The inverse of regularization strength.

**Evaluation Metric:**
**F1-Score** was used as the primary metric on the validation set to compare the performance of different parameter combinations, as it provides a balanced measure of precision and recall, which is crucial for imbalanced classification tasks.

#### 3. Final Training and Prediction

The optimal model identified through GridSearch was then **retrained** on the combined training and validation datasets to maximize the amount of training data. Finally, this fully trained model was used to generate the final predictions on the unseen test dataset.
