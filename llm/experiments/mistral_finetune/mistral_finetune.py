from datasets import load_dataset
from settings import *
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import transformers
from datasets import load_dataset
import numpy as np
import json
from trl import SFTTrainer,get_peft_config, get_peft_config, ModelConfig


model_id = "90stcamp/Mistral_7B_Instruct_v0.2_4_bit" #"mistralai/Mistral-7B-Instruct-v0.2"
with open(API_KEY, "r") as env:
    env_dict = json.load(env)
hf_token = env_dict['huggingface']

cache_dir = "model"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token, cache_dir=cache_dir)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=hf_token, cache_dir=cache_dir,quantization_config=bnb_config, device_map={"":0})
model.resize_token_embeddings(len(tokenizer)) 

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, config)
print_trainable_parameters(model)

# download dataset
dataset = load_dataset("90stcamp/youtube_datasets_v2",token=hf_token)

def get_prompt(topic, text):
    template = f"""<s>[INST] This is the script for {topic} video. Please pick out the scenes or key points that could be the most viewed moments or 'hot clips' from the provided script. [/INST]
        ### Script: {text}
        """
    return template

def merge_columns(example):
    topic = example['topic']
    text = example['text']
    summary = example['summary']
    example["prediction"] = get_prompt(topic, text) + "\nAnswer:" + str(summary) + "<\s>."
    return example


def change_summary(example):
  if example['summary'] == '':
    example['summary'] = 'No Answer'
  return example

selected_columns = ["topic", "text", "summary"]
dataset["train"] = dataset["train"].select_columns(selected_columns)
dataset['train'] = dataset['train'].map(merge_columns)
data = dataset.map(lambda samples: tokenizer(samples['prediction']), batched=True)
data['train'] = data['train'].map(change_summary)


num_epochs = 2
batch_size = 4
gradient_accumulation_steps = 2

total_train_samples = len(data['train'])

total_steps = (total_train_samples // batch_size) // gradient_accumulation_steps * num_epochs

trainer = transformers.Trainer(
    model=model,
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_steps=total_steps,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)


model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
MODEL_SAVE_REPO = '90stcamp/Mistral_7B_Instruct_v0.2_4_bit_finetune_ver3' # ex) 'my-bert-fine-tuned'

## Push to huggingface-hub
model.push_to_hub(
			MODEL_SAVE_REPO, 
			use_temp_dir=True, 
			use_auth_token=hf_token
)
tokenizer.push_to_hub(
			MODEL_SAVE_REPO, 
			use_temp_dir=True, 
			use_auth_token=hf_token
)
