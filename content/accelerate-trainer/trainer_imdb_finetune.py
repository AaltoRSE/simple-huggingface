import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer


def main():
    """
    Main function that does the training

    Based on https://github.com/csc-training/intro-to-dl/blob/master/day1/04b-pytorch-imdb-huggingface.ipynb
    """

    imdb = load_dataset("imdb")

    train_dataset = imdb["train"].shuffle(seed=0).take(20000)
    test_dataset = imdb["test"]

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def preprocess_function(examples):
       return tokenizer(examples["text"], truncation=True)
 
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    def compute_metrics(eval_pred):
       eval_accuracy = evaluate.load("accuracy")
       eval_f1 = evaluate.load("f1")

       logits, labels = eval_pred
       predictions = np.argmax(logits, axis=-1)
       accuracy = eval_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
       f1 = eval_f1.compute(predictions=predictions, references=labels)["f1"]
       return {"accuracy": accuracy, "f1": f1}

    repo_name = "finetuning-sentiment-model-5000-samples"
 
    training_args = TrainingArguments(
       output_dir=repo_name,
       learning_rate=2e-5,
       per_device_train_batch_size=16,
       per_device_eval_batch_size=16,
       num_train_epochs=2,
       weight_decay=0.01,
       save_strategy="epoch",
       push_to_hub=False,
       report_to="none"
    )

    
    trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=tokenized_train,
       eval_dataset=tokenized_test,
       tokenizer=tokenizer,
       data_collator=data_collator,
       compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
