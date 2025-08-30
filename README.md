# hack_llm

Hallucination Detection with BiLSTM + Attention and LRP Explainability

This project implements a deep learning pipeline for detecting hallucinations in ChatGPT responses. It combines a BiLSTM with Multihead Attention for classification and Layer-wise Relevance Propagation (LRP) for interpretability.

📂 Contents

ls_tm.ipynb – End-to-end training pipeline:

Data preprocessing (general_data.json)

Tokenization & padding (Keras Tokenizer)

BiLSTM + Multihead Attention model (PyTorch)

Weighted loss to handle class imbalance

Early stopping, learning rate scheduler

Evaluation: accuracy, F1, confusion matrix, classification report

Saves best model (best_model.pth)

lwrp.ipynb – Model explainability:

Wraps the trained BiLSTM model with LRP / gradient-based attribution

Supports:

Vanilla Gradients

Integrated Gradients

Simplified LRP approximation

Produces token-level relevance heatmaps

Helps analyze which words contributed most to hallucination predictions

🚀 Getting Started
Prerequisites

Install dependencies:

pip install torch torchvision torchaudio
pip install scikit-learn matplotlib numpy pandas
pip install tensorflow  # for keras.preprocessing.text

Running the Models

Prepare the dataset
Place your general_data.json file in the root directory.
Each entry should contain:

{
  "user_query": "...",
  "chatgpt_response": "...",
  "hallucination": "yes/no"
}


Train the model
Open and run ls_tm.ipynb.
The best model will be saved as best_model.pth.

Run explainability
Open and run lwrp.ipynb to analyze predictions using LRP / gradient-based attributions.

📊 Results

Classification Metrics: Accuracy, F1-score, confusion matrix

Visualization: Training history plots (loss, accuracy, F1)

Explainability: Token-level relevance highlighting important words for predictions

🛠️ Customization

Change hyperparameters (embed_dim, hidden_dim, dropout, etc.) in the BiLSTM model.

Replace general_data.json with your own dataset of query–response pairs.

Extend UniversalLRPExplainer to support new explanation methods.
