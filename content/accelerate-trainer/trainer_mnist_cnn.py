import time
import torch
from torch import nn
import transformers

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from transformers import Trainer, TrainingArguments

# Describe model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding='valid'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(32*13*13, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, images=None, **kwargs):
        return self.layers(images)

def main():
    """
    main function that does the training
    """

    data_dir = './data'

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=ToTensor())
    test_dataset = datasets.MNIST(data_dir, train=False, transform=ToTensor())

    model = SimpleCNN()

    def collator_fn(data):
        images = torch.stack([d[0] for d in data])
        labels = torch.tensor([d[1] for d in data])
        return {"images":images, "labels":labels}


    class MNISTTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            images = inputs.pop('images')
            target = inputs.pop('labels')
            output = model(images)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)
            return (loss, outputs) if return_outputs else loss


    trainer_args = TrainingArguments(
        report_to="none",
        num_train_epochs=4,
        eval_strategy="epoch",
        logging_steps=0.1,
    )

    trainer = MNISTTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collator_fn,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
