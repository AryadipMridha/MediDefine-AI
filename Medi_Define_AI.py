# -*- coding: utf-8 -*-
"""MediDefine-AI: QLoRA fine-tuning of Llama-2 on medical term definitions.

Script export of Medi_Define_AI.ipynb, which was built in Google Colab and
adapted from a Hugging Face LLM fine-tuning course notebook.

Requires a CUDA GPU. Install the pinned dependencies first:

    pip install -r requirements.txt

# Fine-Tuning LLMs with Hugging Face

## Step 1: Installing and importing the libraries
"""

import torch
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from transformers import (AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,TrainingArguments,pipeline)

"""## Step 2: Loading the model"""

model=AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="aboonaji/llama2finetune-v2",
   quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=getattr(torch,"float16"),bnb_4bit_quant_type="nf4")

)
model.config.use_cache=False
model.config.pretraining_tp=1

"""## Step 3: Loading the tokenizer"""

tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path= "aboonaji/llama2finetune-v2" ,trust_remote_code=True)
tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side="right"

"""## Step 4: Setting the training arguments"""

training_arguments=TrainingArguments(output_dir="./results",per_device_train_batch_size=4,max_steps=100)

"""## Step 5: Creating the Supervised Fine-Tuning trainer"""

sft_trainer=SFTTrainer(model=model,
                       args=training_arguments,
                       train_dataset=load_dataset(path="aboonaji/wiki_medical_terms_llam2_format",split="train"),
                       tokenizer=tokenizer,
                       peft_config= LoraConfig(r=64,lora_alpha=16,lora_dropout=0.1,task_type="CAUSAL_LM"),
                       dataset_text_field="text")

"""## Step 6: Training the model"""

sft_trainer.train()

"""## Step 7: Chatting with the model"""

user_prompt="Please tell me about sclera"
text_generation_pipeline=pipeline(task="text-generation",model=model,tokenizer=tokenizer,max_length=300)
model_answer=text_generation_pipeline(f"<s>[INST] {user_prompt} [/INST]")
print(model_answer[0]['generated_text'])
