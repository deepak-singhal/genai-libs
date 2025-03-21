# IMPORTS
import os
import openai
import gradio as gr
from dotenv import load_dotenv

# ENVIRONMENT SETUP
MODEL_GPT="gpt-4"
load_dotenv(override=True)
openai.api_type=os.getenv('OPENAI_API_TYPE')
openai.azure_endpoint=os.getenv('OPENAI_API_BASE')
openai.api_version=os.getenv('OPENAI_API_VERSION')
openai.api_key=os.getenv('OPENAI_API_KEY')


# BUSINESS
# Message structure will look like this in chat conversation
# [
#     {"role": "system", "content": "system message here"},
#     {"role": "user", "content": "first user prompt here"},
#     {"role": "assistant", "content": "the assistant's response"},
#     {"role": "user", "content": "the new user prompt"},
# ]
# AND function chat(message, history) 
# message -> "current message"
# history -> [["user message1", "assistant response1"],
#             ["user message2", "assistant response2"],
#             ["user message3", "assistant response3"],]

def chat(message, history):
    messages = [{"role": "system", "content": "You are a helpful assistant"}]

    # Transform history into the correct format
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": message})

    print("\nHistory is : ")
    print(history)
    print("messages is : ")
    print(messages)

    stream = openai.chat.completions.create(model=MODEL_GPT, messages=messages, stream=True)
    response = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content is not None :
            response += chunk.choices[0].delta.content
        yield response

gr.ChatInterface(fn=chat).launch()
