from google.colab import userdata
from huggingface_hub import login
from transformers import AutoTokenizer

hf_token = userdata.get('HF_TOKEN')
login(hf_token)

# Tokenize
text = "I am excited to show Tokenizers in action to my LLM engineers"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokens = tokenizer.encode(text)
print(len(text))
print(tokens)
print(len(tokens))

# De-Tokenize
print(tokenizer.decode(tokens))
print(tokenizer.batch_decode(tokens))
# print(tokenizer.vocab)
# print(tokenizer.get_added_vocab)


############### NOW TRY WITH TRAINED MODEL - INSTRUCT

# Tokenize
text = "I am excited to show Tokenizers in action to my LLM engineers"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
  ]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)

# OUTPUT WILL BE LIKE THIS

  # <|begin_of_text|>
  #   <|start_header_id|>system<|end_header_id|>
  #   Cutting Knowledge Date: December 2023
  #   Today Date: 26 Jul 2024
  #   You are a helpful assistant

  # <|eot_id|>
  #   <|start_header_id|>user<|end_header_id|>
  #   Tell a light-hearted joke for a room of Data Scientists

  # <|eot_id|>
  #   <|start_header_id|>assistant<|end_header_id|>
