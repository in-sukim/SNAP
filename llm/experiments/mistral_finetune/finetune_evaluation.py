from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch 
from settings import * 
import json 
from langchain.text_splitter import CharacterTextSplitter
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
import re
from contextlib import contextmanager
from tqdm import tqdm
from contextlib import contextmanager

import time
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

with open(API_KEY, "r") as env:
    env_dict = json.load(env)
hf_token = env_dict['huggingface']

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

model_checkpoint = "90stcamp/Mistral_7B_Instruct_v0.2_4_bit"
base_model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint, 
    token = hf_token,
    torch_dtype=torch.bfloat16,  # you may change it with different models
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 is recommended
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type='nf4',
    ),
)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, token = hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
ft_model = PeftModel.from_pretrained(base_model, "90stcamp/Mistral_7B_Instruct_v0.2_4_bit_finetune_ver3", token = hf_token )



def get_prompt(topic, text):
    template = f"""<s>[INST] This is the script for {topic} video. Please pick out the scenes or key points that could be the most viewed moments or 'hot clips' from the provided script. [/INST]
        ### Script: {text}
        """
    return template


def merge_columns_for_eval(example):
    url = example['url']
    topic = example['topic']
    text = example['text']
    summary = example['summary']
    example["prediction"] = get_prompt(topic, text)+ "\nAnswer:"+ '<\s>'
    return example

eval_data = np.load('./text_sum_eval_data.npy', allow_pickle=True)


# eval_data
# {doc1}
# {doc2} 
final_json = []
for example in tqdm(eval_data, desc = "Evaluation Datasets Summarization...", total = len(eval_data)):
    # example: {'topic': ~, 'text': ~, 'summary': ~}
    torch.cuda.empty_cache()
    url = example['url']
    topic = example['topic']
    text = example['text']
    summary = example['summary']

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator= '\n',
        chunk_size = 1024,
        chunk_overlap = 0,
        )
    split_docs = text_splitter.create_documents([text])
    # Text chunk(1024 token)
    doc_result = []
    with timer("Chunk Summarization"):
        for doc in split_docs:
            prompt = get_prompt(topic, doc.page_content) + '\nAnswer:'
            ft_model.eval()
            with torch.no_grad():
                model_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
                model_gen = ft_model.generate(**model_input, max_new_tokens=60, repetition_penalty=0.9,pad_token_id=tokenizer.eos_token_id)[0]
                result = tokenizer.decode(model_gen, skip_special_tokens=True)
                print(result)
                # print(result.split("Answer:")[1])
            # pattern = re.compile(r"\[\[.*?\]\]")
            #match = pattern.search(result)
            pattern = re.compile(r"\[\d+\.\d+, \d+\.\d+\]")
            matches = pattern.findall(result)
            print(matches)
            if matches:
                all_result = []
                for i in matches:
                    if eval(i) not in all_result:
                        all_result.append(eval(i))
                text_pattern = re.compile(r"\((\d+\.\d+)\) (.+?)(?=\n \(\d|\Z)", re.DOTALL)
                sentences_with_times = text_pattern.findall(text)

                final_result = []
                for start, end in all_result:
                    concat_sentence = ''
                    for timestamp , sentence in sentences_with_times:
                        if (timestamp == str(start)) or (timestamp == str(end)):
                            concat_sentence += f"\n({timestamp}) " + sentence
                    if concat_sentence != '':
                        final_result.append(concat_sentence.strip())
                print(final_result)
                final_json.append({'url': url, 'topic': topic, 'text': text, 'summary': summary, 'predict':final_result})
            else:
                final_json.append({'url': url, 'topic': topic, 'text': text, 'summary': summary, 'predict':"No Anaswer"})

           # print("Match part:",match)
           # print()
           # print('-'* 100)
           # if match:
                #extracted_text = match.group()
                #extracted_text = eval(extracted_text)
                #print(extracted_text)
                #doc_result.extend(extracted_text)
                #final_json.append({'url': url, 'topic': topic, 'text': text, 'summary': summary, 'predict':doc_result})
            #else:
                #extracted_text = "No Answer"
            # [[123,123],[123,123]]
                #final_json.append({'url': url, 'topic': topic, 'text': text, 'summary': summary, 'predict':extracted_text})
            #{'url':~, 'topic': ~, 'text': ~, 'summary': ~, 'predict': [[123,123],[123,123],[123,123],[123,123],[123,123]]}

new_data = np.array(final_json)
np.save('text_sum_eval_data_finetune.npy', new_data)
