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
    PeftModel,
    TaskType
    )
import torch

checkpoint = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )

model = AutoModelForCausalLM.from_pretrained(
                checkpoint, 
                quantization_config=bnb_config,
                device_map="auto"
            )

model = PeftModel.from_pretrained(model, "clone_peft")
tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='left')


def generate_sample(
    prompt,
    num_return_sequences=1,
    max_new_tokens=24,
    max_length=42
):
    input_ids = tokenizer(
        prompt,
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(model.device)  

    tokens = model.generate(
        input_ids=input_ids,
        max_length=max_length + max_new_tokens,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=0.5,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded_tokens = tokenizer.decode(
        tokens[0],
        padding_side='left',
        skip_special_tokens=True
    )

    if "user" in decoded_tokens:
        response = decoded_tokens.rsplit("user", 1)[-1].strip()
    else:
        response = decoded_tokens.strip()

    return response


# prompt = "<s>пойдешь в дискорд?</s>"
# generated_sample = generate_sample(prompt)
# print(generated_sample)