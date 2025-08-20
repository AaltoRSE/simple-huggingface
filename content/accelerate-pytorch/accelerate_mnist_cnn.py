import time
import torch
from torch import nn
import transformers

from transformers import get_linear_schedule_with_warmup

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from accelerate import Accelerator
import logging

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

    def forward(self, x):
        return self.layers(x)

def main():
    """
    main function that does the training
    """
    
    data_dir = './data'

    n_epochs = 8
    batch_size = 32

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # Define data sets and data loaders
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=ToTensor())
    test_dataset = datasets.MNIST(data_dir, train=False, transform=ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    # Define loss function
    loss_function = nn.CrossEntropyLoss()

    # Define model
    model = SimpleCNN()

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters())

    num_training_steps = n_epochs * len(train_dataloader)

    # Define learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(500, num_training_steps // 10),  # 10% warmup
        num_training_steps=num_training_steps
    )
    
    accelerator = Accelerator()

    device = accelerator.device
    
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    model.train()

    total_steps = 0
    start_time = time.time()

    for epoch in range(n_epochs):

        epoch_loss = 0
        epoch_steps = 0
        
        if accelerator.is_main_process:
            logger.info(f"Starting epoch {epoch + 1}/{n_epochs}")
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            
            total_steps += 1
            epoch_loss += loss.item()
            epoch_steps += 1

            # Enhanced logging
            if total_steps % 100 == 0 and accelerator.is_main_process:
                elapsed_time = time.time() - start_time
                avg_loss = epoch_loss / epoch_steps
                current_lr = scheduler.get_last_lr()[0]
                steps_per_sec = total_steps / elapsed_time
                
                logger.info(
                    f"Step {total_steps} | Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                    f"Steps/sec: {steps_per_sec:.2f}"
                )
                
    accelerator.end_training()


if __name__ == "__main__":
    main()
    #torch.distributed.destroy_process_group()
