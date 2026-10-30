# 📊Sentiment-Analysis-on-Google-Play-Store-Data-using-NLP


## 📌 Project Overview

**Sentiment Analysis on Google Play Store Data Using NLP** is a machine learning project that analyzes user reviews and classifies them into three sentiment categories: **Positive, Negative, and Neutral**.

The project uses Natural Language Processing (NLP) techniques to preprocess textual reviews and TF-IDF (Term Frequency–Inverse Document Frequency) to convert text into numerical features. Multiple machine learning algorithms are trained and evaluated to analyze their performance in sentiment classification.

This project helps understand user opinions, feedback, and satisfaction toward mobile applications.

## 🎯 Objectives

- Analyze user reviews collected from Google Play Store data.
- Clean and preprocess textual review data.
- Classify reviews into positive, negative, and neutral sentiments.
- Convert text into numerical features using TF-IDF.
- Train and evaluate multiple machine learning models.
- Visualize sentiment distribution and review statistics.
- Save trained models for future use.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Natural Language Processing:** NLTK
- **Feature Extraction:** TF-IDF Vectorizer
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn
- **Model Serialization:** Pickle
- **Development Environment:** Jupyter Notebook

## 📂 Project Structure

Sentiment-Analysis-Google-Play-Store/
│
├── main_sentiment.ipynb
├── reviews.csv
├── tfidf_vectorizer.pkl
├── RandomForest_model.pkl
├── DecisionTree_model.pkl
├── MLP_model.pkl
├── sentiment_distribution.png
├── data_visualizations.png
└── README.md


## 📊 Dataset

The project uses a CSV dataset named `reviews.csv`.

The notebook selects the following columns:

- `content` — Text of the user review.
- `score` — Numerical rating associated with the review.

The numerical rating is used to assign sentiment labels.

### Sentiment Classification

| Rating | Sentiment |
|---|---|
| 4–5 | Positive |
| 3 | Neutral |
| Below 3 | Negative |

The dataset is cleaned by removing rows with missing values before the sentiment labels are generated.

**Dataset requirement:** Place `reviews.csv` in the same directory as the notebook before running the project.

## ⚙️ Project Workflow

### 1. Data Loading

The dataset is loaded using Pandas. The relevant review text and rating columns are selected for analysis.

### 2. Data Cleaning

The preprocessing stage includes:

- Converting text to lowercase.
- Removing special characters and numbers.
- Tokenizing text into individual words.
- Removing English stopwords.
- Removing records with missing review or rating values.

These steps prepare the review text for feature extraction.

### 3. Sentiment Labeling

Sentiment labels are generated based on the numerical ratings:

- **Positive:** Rating greater than or equal to 4.
- **Neutral:** Rating equal to 3.
- **Negative:** Rating below 3.

### 4. Feature Extraction Using TF-IDF

The `TfidfVectorizer` converts cleaned review text into numerical features that machine learning models can process.

The notebook uses the following configuration:

- Maximum features: `10,000`
- N-gram range: `(1, 2)`

This configuration considers individual words and pairs of consecutive words.

### 5. Data Visualization

Matplotlib and Seaborn are used to explore the dataset and visualize:

- Sentiment distribution.
- Rating distribution.
- Rating statistics.
- Sentiment counts.
- Relationship between ratings and sentiment categories.

### 6. Model Training

The dataset is split into training and testing sets using an 80:20 ratio.

Three machine learning models are trained and evaluated:

| Algorithm | Configuration |
|---|---|
| Random Forest Classifier | 200 estimators |
| Decision Tree Classifier | Maximum depth of 20 |
| Multi-Layer Perceptron (MLP) | Hidden layers: 100 and 50 neurons |

A random state of 42 is used for reproducibility in the train-test split and model configurations.

### 7. Model Evaluation

The trained models are evaluated using:

- **Accuracy Score:** Measures the proportion of correct predictions.
- **Classification Report:** Provides precision, recall, F1-score, and support for each sentiment class.

The trained models are saved as Pickle files for potential reuse.

## 📈 Results

The notebook prints the accuracy score and classification report for each trained model.

The actual performance depends on the dataset and the execution results.

**Note:** Add the measured accuracy values here after running the notebook. No accuracy figures are included because they should reflect the actual experimental results.

| Model | Accuracy |
|---|---|
| Random Forest | Add measured result |
| Decision Tree | Add measured result |
| MLP | Add measured result |

## 🚀 Installation and Setup

### Prerequisites

Install Python and Jupyter Notebook on your system.

### Step 1: Clone the Repository

```bash
git clone <your-github-repository-url>
```

### Step 2: Navigate to the Project Directory

```bash
cd Sentiment-Analysis-Google-Play-Store
```

### Step 3: Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn jupyter
```

### Step 4: Add the Dataset

Place the `reviews.csv` dataset in the project directory.

### Step 5: Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `main_sentiment.ipynb` and execute the notebook cells in order.

The notebook downloads the required NLTK resources during preprocessing.

## 💡 Applications

This project demonstrates techniques that can be applied to:

- Understanding customer feedback.
- Analyzing mobile application reviews.
- Identifying user satisfaction trends.
- Supporting application improvement decisions.
- Exploring public opinions about digital products.

## 🔮 Future Enhancements

- Develop a web application using Flask.
- Allow users to enter reviews and receive sentiment predictions.
- Compare traditional machine learning models with transformer-based NLP models.
- Improve text preprocessing to handle emojis, negation, and informal language.
- Evaluate models using additional metrics and cross-validation.
- Deploy the application for online access.

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

- Text preprocessing and Natural Language Processing.
- TF-IDF feature extraction.
- Training machine learning classification models.
- Evaluating model performance.
- Visualizing data using Python libraries.
- Saving trained models using Pickle.
