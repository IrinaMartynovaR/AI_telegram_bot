import pandas as pd
from datasets import Dataset, load_from_disk
import re
import json

def process_chats(file_path: str):
    df = pd.read_json(file_path)

    messages = []
    for sample in df["chats"]["list"]:
        for row in sample["messages"]:
            if row["text"] != '':
                username = row['from']
                if username != "Ваше_имя_Пользователя":#исходя из переписки
                    username = "User"
                if username == "Ваше_имя_Пользователя":
                    username = "Clone"  
                message = f"{username}: {row['text']}"
                messages.append(message)
    
    return messages
c = process_chats('res_id.json')


merged_messages = []
current_user = ''

for message in c:
    if message.startswith('User:'):
        if current_user != 'User':
            current_user = 'User'
            merged_messages.append(message)
        else:
            merged_messages[-1] += '\n' + message[len('User: '):]
    else:
        if current_user != 'Clone':
            current_user = 'Clone'
            merged_messages.append(message)
        else:
            merged_messages[-1] += '\n' + message[len('Clone: '):]


pattern = r"\{'type': 'link', 'text': '.*?'\}"
clean_url = [re.sub(pattern, "", msg) for msg in merged_messages]
pattern = r"\{'type': 'mention', 'text': '.*?'\}"
clean_mention = [re.sub(pattern, "", msg) for msg in clean_url]
pattern = r'\[.*?\]'
result_messages = [re.sub(pattern, " ", url) for url in clean_mention]

merged_messages = clean_mention


def add_tokens(messages):
    result = []
    for message in messages:
        speaker, text = message.split(': ', 1)
        if speaker == 'User':
            result.append("<s>user\n" + text + "</s>")
        elif speaker == 'Clone':
            result.append("<s>bot\n" + text + "</s>")
    return result

# Разделение и добавление служебных токенов
processed_messages = add_tokens(merged_messages)


merged_messages = processed_messages

size = 5
num_steps = len(merged_messages)/5
samples = ("\n".join(merged_messages[i*size:(i+1)*size]) for i in range(round(num_steps)))

df = pd.DataFrame({"prompt": samples})
dataset = Dataset.from_pandas(df)
dataset.save_to_disk("clon_conversations")

print(dataset)


from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
    )
import torch


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)



checkpoint = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    quantization_config=bnb_config,
    device_map="auto"
)


model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


print(tokenizer.bos_token_id)
print(tokenizer.eos_token_id)
print(tokenizer.decode([1]))
print(tokenizer.decode([2]))


def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(prompt, return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()


def get_message(inputs):
    conversation = Conversation()
    conversation.add_user_message(inputs)
    prompt = conversation.get_prompt(tokenizer)
    print('Промт', '\n', '*'*100)
    print(prompt)
    print('*'*100)
    output = generate(model, tokenizer, prompt, generation_config)
    return output


peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
              "q_proj", 
              "k_proj",
              "v_proj",
              "o_proj"],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()



dataset = load_from_disk("clon_conversations")
dataset = dataset.map(lambda example: tokenizer(example["prompt"], max_length=256), batched=True)
dataset = dataset.train_test_split(0.1, 0.9)


collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="llama",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    logging_steps=100,
    save_steps=1000,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    fp16=True,
    num_train_epochs=14,#недостаточно эпох для качественного обучения
    ddp_find_unused_parameters=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()

model.save_pretrained("clone_peft")