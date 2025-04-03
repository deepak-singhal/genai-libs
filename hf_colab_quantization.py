import torch
from google.colab import userdata
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig

# Login to HuggingFace with Token stored in Google Colab config
hf_token = userdata.get("HF_TOKEN")
login(hf_token)

# LLM Models - Instruct
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PHI3 = "microsoft/Phi-3-mini-4k-instruct"
GEMMA2 = "google/gemma-2-2b-it"
QWEN2 = "Qwen/Qwen2-7B-Instruct" # exercise for you
MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1" # If this doesn't fit it your GPU memory, try others from the hub

# Messages
messages = [
    {"role":"system", "content":"You are an helpful assistant"},
    {"role":"user", "content":"Tell a light-hearted joke"}
]

# Quantization Config - this allows us to load the model into memory and use less memory
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

def generate(model, messages):
  # Tokenizer
  tokenizer = AutoTokenizer.from_pretrained(model)
  tokenizer.pad_token = tokenizer.eos_token
  inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
  streamer = TextStreamer(tokenizer)
  # Model
  model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", quantization_config=quant_config)
  # Generate Output
  outputs = model.generate(inputs, max_new_tokens=80, streamer=streamer)
  # CleanUp
  del tokenizer, streamer, model, inputs, outputs
  torch.cuda.empty_cache()

generate(LLAMA, messages)
# generate(PHI3, messages)
# generate(GEMMA2, messages)
# generate(QWEN2, messages)
# generate(MIXTRAL, messages)
