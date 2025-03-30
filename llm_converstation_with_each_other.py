# IMPORTS
import os
import openai
import ollama
from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display

# CONSTANTS
MODEL_GPT="gpt-4"
MODEL_LLAMA="llama3.2"

# ENVIRONMENT SETUP
load_dotenv(override=True)
openai.api_type=os.getenv('OPENAI_API_TYPE')
openai.azure_endpoint=os.getenv('OPENAI_API_BASE')
openai.api_version=os.getenv('OPENAI_API_VERSION')
openai.api_key=os.getenv('OPENAI_API_KEY')

#PROMPTS
gpt_system = "You are a chatbot who is very argumentative; \
you disagree with anything in the conversation and you challenge everything, in a snarky way."

ollama_system = "You are a very polite, courteous chatbot. You try to agree with \
everything the other person says, or find common ground. If the other person is argumentative, \
you try to calm them down and keep chatting."


# SYSTEM : GPT       ASSISTANT : GPT        USER : OLLAMA
def call_gpt():
    messages = [{"role": "system", "content": gpt_system}]
    for gpt, ollama_message in zip(gpt_messages, ollama_messages):
        messages.append({"role": "assistant", "content": gpt})
        messages.append({"role": "user", "content": ollama_message})
    response = openai.chat.completions.create(
        model=MODEL_GPT,
        messages=messages
    )
    return response.choices[0].message.content


# SYSTEM : OLLAMA    ASSISTANT : OLLAMA     USER : GPT
def call_ollama():
    messages = [{"role":"system", "content":ollama_system}]
    for gpt, ollama_message in zip(gpt_messages, ollama_messages):
        messages.append({"role": "assistant", "content": ollama_message})
        messages.append({"role": "user", "content": gpt})
    response = ollama.chat(model=MODEL_LLAMA, messages=messages)
    return response['message']['content']



# MAIN FLOW 
gpt_messages = ["Hi there"]
ollama_messages = ["Hi"]

print(f"GPT:\n{gpt_messages[0]}\n")
print(f"Ollama:\n{ollama_messages[0]}\n")

for i in range(3):
    gpt_next = call_gpt()
    print(f"GPT:\n{gpt_next}\n")
    gpt_messages.append(gpt_next)
    
    ollama_next = call_ollama()
    print(f"Ollama:\n{ollama_next}\n")
    ollama_messages.append(ollama_next)

    
