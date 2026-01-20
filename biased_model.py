import re
import os
import sys
import glob
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# -------------------- Text Cleaning --------------------
def clean_text(text):
    """Clean and normalize text by removing extra whitespace and line breaks."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n', ' ', text)
    return text.strip()

# -------------------- Text Chunking --------------------
def split_text(text, max_length=512):
    """Split long text into chunks with max_length words for model processing."""
    words = text.split()
    chunks, current_chunk = [], []

    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) > max_length:
            chunks.append(" ".join(current_chunk[:-1]))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# -------------------- Initialize Models --------------------
# Sentiment pipeline
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME)

# Zero-shot classifier for bias detection (used only if not using your trained model)
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# -------------------- Sentiment Analysis --------------------
def get_final_sentiment(text, sentiment_pipe=sentiment_pipeline):
    LABEL_MAP = {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "NEUTRAL",
        "LABEL_2": "POSITIVE"
    }

    cleaned_text = clean_text(text)
    chunks = split_text(cleaned_text)
    sentiment_scores = []

    for chunk in chunks:
        try:
            result = sentiment_pipe(chunk)
            raw_label = result[0]['label']
            readable_label = LABEL_MAP.get(raw_label, raw_label)
            sentiment_scores.append(readable_label)
        except Exception as e:
            print("Sentiment analysis failed on chunk:", e)

    if sentiment_scores:
        final_sentiment = max(set(sentiment_scores), key=sentiment_scores.count)
        return final_sentiment
    return "Unknown"

# -------------------- Bias Detection (Zero-Shot fallback) --------------------
def detect_bias_zero_shot(text):
    candidate_labels = ["left bias", "right bias", "neutral"]
    cleaned_text = clean_text(text)
    try:
        result = zero_shot_classifier(cleaned_text, candidate_labels)
        bias_scores = {label: round(score, 4) for label, score in zip(result["labels"], result["scores"])}
        return bias_scores
    except Exception as e:
        print("Bias detection failed:", e)
        return {}

# -------------------- Load Dataset --------------------
def load_parquet_files_from_folder(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "*.parquet"))
    df_list = []
    for file in all_files:
        df_list.append(pd.read_parquet(file))
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

# -------------------- Train Model --------------------
def train_model(data_df):
    print("Preparing to train model...")

    # Check if 'bias_label' column exists and is int type
    if 'bias_label' not in data_df.columns:
        print("ERROR: Your dataset must contain a 'bias_label' column with integer labels for bias categories.")
        return

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(data_df['bias_label'].unique()))

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples['article'], padding="max_length", truncation=True, max_length=512)

    dataset = Dataset.from_pandas(data_df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Rename label column to 'labels' for Trainer compatibility
    tokenized_dataset = tokenized_dataset.rename_column("bias_label", "labels")
    tokenized_dataset.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        # Add eval_dataset here if you have validation data
    )

    # Train!
    trainer.train()
    trainer.save_model("./bias_model")
    print("Training complete and model saved at ./bias_model")

# -------------------- Main Entry --------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python biased_model.py train <training_data_folder>")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "train":
        if len(sys.argv) < 3:
            print("Please provide training data folder path containing parquet files.")
            sys.exit(1)

        training_folder = sys.argv[2]
        print(f"Loading training data from: {training_folder}")

        data_df = load_parquet_files_from_folder(training_folder)
        print(f"Loaded {len(data_df)} records.")
        print("Sample data preview:")
        print(data_df.head())

        # Make sure 'bias_label' column exists and is prepared in your parquet files before training!
        train_model(data_df)

    else:
        print(f"Unknown mode: {mode}")
        print("Supported modes: train")
