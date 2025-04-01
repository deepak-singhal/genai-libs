from google.colab import userdata
from huggingface_hub import login
from transformers import AutoTokenizer

hf_token = userdata.get('HF_TOKEN')
login(hf_token)

LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PHI3_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
QWEN2_MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
STARCODER2_MODEL_NAME = "bigcode/starcoder2-3b"

llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
phi_tokenizer = AutoTokenizer.from_pretrained(PHI3_MODEL_NAME)
qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN2_MODEL_NAME)
starcoder_tokenizer = AutoTokenizer.from_pretrained(STARCODER2_MODEL_NAME)


# Sample text for simple tokernization
text = "Tell a light-hearted joke for a room of Data Scientists"

# Sample message for chat template and tokenization
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": text}
  ]

###################################################### LLAMA ######################################################
# Tokenize
tokens = llama_tokenizer.encode(text)
# print(len(text))
# print(tokens)
# print(len(tokens))

# De-Tokenize
# print(llama_tokenizer.decode(tokens))
print(llama_tokenizer.batch_decode(tokens))
# print(llama_tokenizer.vocab)
# print(llama_tokenizer.get_added_vocab)

# Chat Template
prompt = llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)

###################################################### PHI3 ######################################################
# Tokenize
tokens = phi_tokenizer.encode(text)
# print(len(text))
# print(tokens)
# print(len(tokens))

# De-Tokenize
# print(phi_tokenizer.decode(tokens))
# print(phi_tokenizer.batch_decode(tokens))
# print(phi_tokenizer.vocab)
# print(phi_tokenizer.get_added_vocab)

# Chat Template
prompt = phi_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# print(prompt)

###################################################### QWEN ######################################################
# Tokenize
tokens = qwen_tokenizer.encode(text)
# print(len(text))
# print(tokens)
# print(len(tokens))

# De-Tokenize
# print(qwen_tokenizer.decode(tokens))
# print(qwen_tokenizer.batch_decode(tokens))
# print(qwen_tokenizer.vocab)
# print(qwen_tokenizer.get_added_vocab)

# Chat Template
prompt = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# print(prompt)

###################################################### STARCODER ######################################################
# This is a code generation model
# Tokenize
code = """
def hello_world(person):
  print("Hello", person)
"""
tokens = starcoder_tokenizer.encode(code)
# print(len(code))
# print(tokens)
# print(len(tokens))

# De-Tokenize
# print(starcoder_tokenizer.decode(tokens))
# print(starcoder_tokenizer.batch_decode(tokens))
# print(starcoder_tokenizer.vocab)
# print(starcoder_tokenizer.get_added_vocab)
