# IMPORTS
import os
import openai
import ollama
import gradio as gr
from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display


# ENVIRONMENT SETUP
MODEL_GPT="gpt-4"
MODEL_OLLAMA="llama3.2"
load_dotenv(override=True)
openai.api_type=os.getenv('OPENAI_API_TYPE')
openai.azure_endpoint=os.getenv('OPENAI_API_BASE')
openai.api_version=os.getenv('OPENAI_API_VERSION')
openai.api_key=os.getenv('OPENAI_API_KEY')


# BUSINESS
def message_gpt(prompt):
    messages = [{"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=MODEL_GPT,
        messages=messages
    )
    return response.choices[0].message.content


def message_gpt_stream(prompt):
    messages = [{"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}]
    stream = openai.chat.completions.create(
        model=MODEL_GPT,
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or '' if len(chunk.choices)>0 else ''
        yield result


def message_ollama(prompt):
    messages = [{"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}]
    response = ollama.chat(model=MODEL_OLLAMA, messages=messages)
    return response['message']['content']


def select_model(prompt, model_name):
    if model_name=="GPT":
        result=message_gpt(prompt)
    elif model_name=="OLLAMA":
        result=message_ollama(prompt)
    else:
        raise ValueError("Can Not Find Model")
    return result


# UI MARKDOWN
gr.Interface(
    fn=select_model,
    inputs=[gr.Textbox(label="Your message:"),
            gr.Dropdown(["GPT","OLLAMA","OTHERS"], label="Select LLM Model")           ],
    outputs=[gr.Markdown(label="Response:")],
    flagging_mode="never").launch()
