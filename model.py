from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model,  prepare_model_for_int8_training


from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name_or_path = "ai-forever/rugpt3medium_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(DEVICE)


config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["wte", "lm_head"]
)

config.inference_mode = False

model.enable_input_require_grads()
model = get_peft_model(model, config)
model = prepare_model_for_int8_training(model)


