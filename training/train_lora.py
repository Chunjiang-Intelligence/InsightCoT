import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model with InsightCoT using LoRA.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3-8B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--dataset_name", type=str, default="synthetic_data.jsonl", help="Path to the JSONL dataset")
    parser.add_argument("--output_dir", type=str, default="./results/default_run", help="Directory to save training results")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Per-device training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    args = parser.parse_args()

    model_name = args.model_name
    dataset_name = args.dataset_name
    output_dir = args.output_dir
    num_train_epochs = args.num_train_epochs
    per_device_train_batch_size = args.per_device_train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    learning_rate = args.learning_rate
    max_seq_length = args.max_seq_length
    lora_r = 16
    lora_alpha = 32

    print(f"Loading dataset from {dataset_name}...")
    def format_instruction(sample):
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{sample['Q']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{sample['A']}<|eot_id|>"
    dataset = load_dataset("json", data_files=dataset_name, split="train")
    dataset = dataset.map(lambda sample: {"text": format_instruction(sample)})
    print(f"Dataset loaded and formatted. Number of samples: {len(dataset)}")
    
    print(f"Loading base model: {model_name}...")
    use_4bit = True
    bnb_4bit_compute_dtype = "bfloat16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = False
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        logging_steps=25,
        learning_rate=learning_rate,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )
    print("Starting training...")
    trainer.train()

    final_model_path = os.path.join(output_dir, "final_model")
    print(f"Training complete. Saving final LoRA adapter to {final_model_path}")
    trainer.model.save_pretrained(final_model_path)
    
    print("Script finished successfully.")


if __name__ == "__main__":
    main()
