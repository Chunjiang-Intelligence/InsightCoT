import os
import torch
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
    model_name = "meta-llama/Llama-3-8B-Instruct" 
    dataset_name = "data/synthetic_data.jsonl"
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = "all-linear" 
    use_4bit = True
    bnb_4bit_compute_dtype = "bfloat16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = False
    output_dir = "./results-llama3-8b-insightcot"
    num_train_epochs = 1
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 2
    learning_rate = 2e-4
    max_grad_norm = 0.3
    warmup_ratio = 0.03
    lr_scheduler_type = "cosine"
    max_seq_length = 25565
    packing = False

    print(f"Loading dataset from {dataset_name}...")
    def format_instruction(sample):
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{sample['Q']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{sample['A']}<|eot_id|>"""

    dataset = load_dataset("json", data_files=dataset_name, split="train")
    dataset = dataset.map(lambda sample: {"text": format_instruction(sample)})
    print(f"Dataset loaded and formatted. Number of samples: {len(dataset)}")
    
    print(f"Loading base model: {model_name}...")
    
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
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
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
        optim="paged_adamw_32bit", # Memory-efficient optimizer
        save_strategy="epoch",
        logging_steps=25,
        learning_rate=learning_rate,
        weight_decay=0.001,
        fp16=False, # Set to True if not using bfloat16
        bf16=True,  # Set to True if your GPU supports it (Ampere or newer)
        max_grad_norm=max_grad_norm,
        max_steps=-1,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
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
        packing=packing,
    )

    print("Starting training...")
    trainer.train()

    final_model_path = os.path.join(output_dir, "final_model")
    print(f"Training complete. Saving final LoRA adapter to {final_model_path}")
    trainer.model.save_pretrained(final_model_path)
    
    print("Script finished successfully.")

if __name__ == "__main__":
    main()
