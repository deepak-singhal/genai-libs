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
