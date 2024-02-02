from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model,  prepare_model_for_int8_training


from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name_or_path = "ai-forever/rugpt3medium_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(DEVICE)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
config.inference_mode = False

model.enable_input_require_grads()
model = get_peft_model(model, config)
model = prepare_model_for_int8_training(model)

from peft import PromptEmbedding, PromptTuningConfig, get_peft_model

config = PromptTuningConfig(
    peft_type="PROMPT_TUNING",
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
    token_dim=512,
    num_transformer_submodules=1,
    num_attention_heads=12,
    num_layers=12,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral",
    tokenizer_name_or_path="ai-forever/rugpt3medium_based_on_gpt2",
)

model.enable_input_require_grads()
model = get_peft_model(model, config)
model.enable_input_require_grads()
