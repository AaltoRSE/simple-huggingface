import os
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError

def create_datasets(tokenizer, dataset_name, dataset_splits, training_args=None, apply_chat_template=False, verbose=False):
    # Since we don't need chat template, remove the preprocess function
    
    raw_datasets = DatasetDict()
    for split in dataset_splits.split(","):
        try:
            # Try first if dataset on a Hub repo
            dataset = load_dataset(dataset_name, split=split)
        except DatasetGenerationError:
            # If not, check local dataset
            dataset = load_from_disk(os.path.join(dataset_name, split))

        if "train" in split:
            raw_datasets["train"] = dataset
        elif "test" in split:
            raw_datasets["test"] = dataset
        else:
            raise ValueError(
                f"Split type {split} not recognized as one of test or train."
            )

    # No chat template processing needed since we only have content column
    
    train_data = raw_datasets["train"]
    valid_data = raw_datasets["test"]
    
    # Only print from main process when verbose=True
    if verbose:
        print(
            f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
        )
        print(f"A sample of train dataset: {train_data[0]}")

    return train_data, valid_data