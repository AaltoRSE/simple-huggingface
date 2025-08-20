import os
import torch
import torch.distributed
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
import logging
import argparse

from peft import LoraConfig, get_peft_model, TaskType
from utils import create_datasets

class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        max_length = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        
        for ids, labs in zip(input_ids, labels):
            padding_length = max_length - len(ids)
            
            # Pad input_ids
            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            padded_input_ids.append(padded_ids)
            
            # Pad labels (use -100 for padding tokens so they're ignored in loss calculation)
            padded_labs = labs + [-100] * padding_length
            padded_labels.append(padded_labs)
            
            # Create attention mask
            attention_mask = [1] * len(ids) + [0] * padding_length
            attention_masks.append(attention_mask)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.bool),
            "labels": torch.tensor(padded_labels, dtype=torch.long)
        }


def parse_args():
    parser = argparse.ArgumentParser(description="FSDP Training with Accelerate")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-70b-chat-hf", help="Model checkpoint to use")
    parser.add_argument("--dataset_name", type=str, default="smangrul/code-chat-assistant-v1", help="Dataset name")
    parser.add_argument("--dataset_splits", type=str, default="train,test", help="Dataset configuration")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--use_quantization", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--mixed_precision", type=str, default="no", help="Mixed precision type")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio")
    
    return parser.parse_args()


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize function for causal language modeling"""
    tokenized = tokenizer(
        examples['content'],
        truncation=True,
        padding=False,
        max_length=max_length,
        return_special_tokens_mask=False,
    )
    
    tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
    
    valid_indices = [i for i, ids in enumerate(tokenized["input_ids"]) if len(ids) >= 10]
    
    if len(valid_indices) < len(tokenized["input_ids"]):
        for key in tokenized.keys():
            tokenized[key] = [tokenized[key][i] for i in valid_indices]
    
    return tokenized


def main():
    args = parse_args()
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_dir=args.output_dir,
    )

    if accelerator.is_main_process:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
            level=logging.INFO,
            force=True
        )
        
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("datasets").setLevel(logging.WARNING)
        logging.getLogger("accelerate").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)
        logging.getLogger("peft").setLevel(logging.WARNING)
    else:
        logging.basicConfig(
            format="%(asctime)s - RANK{} - %(levelname)s - %(message)s".format(accelerator.process_index),
            datefmt="%H:%M:%S",
            level=logging.ERROR,
            force=True
        )
        
        logging.getLogger("transformers").setLevel(logging.CRITICAL)
        logging.getLogger("datasets").setLevel(logging.CRITICAL)
        logging.getLogger("accelerate").setLevel(logging.CRITICAL)
        logging.getLogger("torch").setLevel(logging.CRITICAL)
        logging.getLogger("peft").setLevel(logging.CRITICAL)
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    
    if accelerator.is_main_process:
        logger.info(f"Mixed precision: {accelerator.mixed_precision} | GPUs: {accelerator.num_processes}")
        logger.info(f"Model: {args.model_name}")
        if args.use_lora:
            logger.info(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    if accelerator.is_main_process:
        logger.info(f"Tokenizer loaded, pad_token_id: {tokenizer.pad_token_id}")
    
    if accelerator.mixed_precision == "fp16":
        torch_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
 
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        # device_map={"": "cpu"},
        trust_remote_code=True,
        attn_implementation="eager",
    )
    
    # Ensure no KV caching during training (important with checkpointing/FSDP)
    model.config.use_cache = False
    
    # Setup LoRA if requested
    if args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Common for LLaMA
            # bias="none",  # Explicitly set bias handling
            init_lora_weights=True,  # Ensure proper initialization
        )
        model = get_peft_model(model, lora_config)
        
        # # Initialize LoRA weights properly to prevent NaN
        # for name, module in model.named_modules():
        #     if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
        #         # Re-initialize LoRA weights with smaller values
        #         if hasattr(module.lora_A, 'default'):
        #             torch.nn.init.normal_(module.lora_A.default.weight, std=0.01)
        #         if hasattr(module.lora_B, 'default'):
        #             torch.nn.init.zeros_(module.lora_B.default.weight)
        
        if accelerator.is_main_process:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"LoRA enabled: {trainable_params:,}/{total_params:,} ({100 * trainable_params / total_params:.2f}%) trainable")
    else:
        if accelerator.is_main_process:
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Full fine-tuning: {total_params:,} parameters")
    
    # Ensure all model parameters have consistent dtype for FSDP
    model = model.to(dtype=torch_dtype)
    
    # Load and prepare datasets
    train_dataset, eval_dataset = create_datasets(
        tokenizer,
        args.dataset_name,
        args.dataset_splits,
        verbose=accelerator.is_main_process,
    )
    
    # Tokenize datasets
    def tokenize_fn(examples):
        return tokenize_function(examples, tokenizer, args.max_length)
    
    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )
    
    eval_dataset = eval_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval dataset",
    )
    
    # Dataset info - only from main process
    if accelerator.is_main_process:
        logger.info(f"Dataset: {len(train_dataset)} train, {len(eval_dataset)} eval samples")
    
    data_collator = CustomDataCollator(tokenizer)
    
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=args.batch_size, 
        collate_fn=data_collator,
        num_workers=7,
        pin_memory=True,
        drop_last=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size, 
        collate_fn=data_collator,
        num_workers=7,
        pin_memory=True,
        drop_last=True
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    num_training_steps = args.num_epochs * len(train_dataloader) // args.gradient_accumulation_steps
    
    if args.warmup_ratio > 0:
        warmup_steps = int(args.warmup_ratio * num_training_steps)
    else:
        warmup_steps = args.warmup_steps
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    if accelerator.is_main_process:
        logger.info(f"Training: {num_training_steps} steps, {warmup_steps} warmup, LR={args.learning_rate}")
    
    # Wait for all processes before preparing
    # accelerator.wait_for_everyone()
    
    # Prepare everything with accelerator - with error handling
    try:
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
        if accelerator.is_main_process:
            logger.info("Model and dataloaders prepared ✓")
    except Exception as e:
        logger.error(f"Failed to prepare components: {e}")
        raise
    
    # Initialize tracker
    # if accelerator.is_main_process:
    #     accelerator.init_trackers("fsdp_training")
    
    # Training loop
    total_steps = 0
    model.train()
    
    if accelerator.is_main_process:
        logger.info("=" * 50)
        logger.info("TRAINING STARTED")
        logger.info(f"Epochs: {args.num_epochs} | Batch size: {args.batch_size} | Grad accum: {args.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
        logger.info("=" * 50)
    
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                
                # Check for NaN loss and skip if found
                if torch.isnan(loss) or torch.isinf(loss):
                    if accelerator.is_main_process:
                        logger.warning(f"Skipping step {total_steps} due to NaN/Inf loss")
                    continue
                
                total_loss += loss.detach().float()
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    # Check for NaN gradients before clipping
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    # Skip step if gradient norm is NaN or too large
                    if torch.isnan(grad_norm) or grad_norm > 100.0:
                        if accelerator.is_main_process:
                            logger.warning(f"Skipping step {total_steps} due to large gradient norm: {grad_norm}")
                        optimizer.zero_grad()
                        continue
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                total_steps += 1
                
                # Logging
                if total_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss / args.logging_steps)
                    if accelerator.is_main_process:
                        logger.info(f"Step {total_steps:5d} | Loss: {avg_loss.mean():.4f} | LR: {lr_scheduler.get_last_lr()[0]:.2e}")
                        accelerator.log({
                            "train_loss": avg_loss.mean(),
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "step": total_steps
                        }, step=total_steps)
                    total_loss = 0
                
                # Save checkpoint
                if total_steps % args.save_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{total_steps}")
                    accelerator.save_state(checkpoint_dir)
                    if accelerator.is_main_process:
                        logger.info(f"✓ Checkpoint saved: step {total_steps}")
        
        # Evaluation at the end of each epoch
        model.eval()
        eval_loss = 0
        eval_steps = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                outputs = model(**batch)
                eval_loss += outputs.loss.detach().float()
                eval_steps += 1
        
        # Gather evaluation metrics from all processes
        eval_loss = accelerator.gather_for_metrics(eval_loss / eval_steps)
        
        if accelerator.is_main_process:
            avg_eval_loss = eval_loss.mean().item()
            
            logger.info(f"Epoch {epoch + 1}/{args.num_epochs} | Eval Loss: {avg_eval_loss:.4f}")
            
            accelerator.log({
                "eval_loss": avg_eval_loss,
                "epoch": epoch + 1
            }, step=total_steps)
    
    # Save final model - use accelerator.save_model for proper FSDP handling
    final_dir = os.path.join(args.output_dir, "final_model")
    
    if args.use_lora:
        # For LoRA models, save only the adapters first
        lora_dir = os.path.join(args.output_dir, "lora_adapters")
        
        # Use accelerator.save_model for distributed-safe saving
        accelerator.save_model(model, lora_dir)
        
        if accelerator.is_main_process:
            # Save tokenizer only on main process
            tokenizer.save_pretrained(lora_dir)
            logger.info(f"LoRA adapters saved to {lora_dir}")
            
            # Option 2: Merge and save full model (only on main process after adapters are saved)
            logger.info("Merging LoRA weights with base model...")
            
            # Load the saved LoRA model to merge
            from peft import PeftModel
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch_dtype,
                device_map=accelerator.device,
                trust_remote_code=True,
            )
            lora_model = PeftModel.from_pretrained(base_model, lora_dir)
            merged_model = lora_model.merge_and_unload()
            
            merged_model.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
            logger.info(f"Full merged model saved to {final_dir}")
    else:
        # Regular model saving using accelerator.save_model
        accelerator.save_model(model, final_dir)
        
        if accelerator.is_main_process:
            tokenizer.save_pretrained(final_dir)
            logger.info(f"Training completed. Final model saved to {final_dir}")
    
    # Wait for all processes to complete before ending
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()