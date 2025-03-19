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

# PROMPTS
system_prompt="You are a python expert and teaching me python language"
user_prompt="""
Please explain what this code does and why:
yield from {book.get("author") for book in books if book.get("author")}
"""

# GPT-4 RESPONSE WITH STREAMING
stream = openai.chat.completions.create(
        model=MODEL_GPT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
          ],
        stream=True
    )
    
response = ""
display_handle = display(Markdown(""), display_id=True)
for chunk in stream:
    response += chunk.choices[0].delta.content or '' if len(chunk.choices)>0 else ''
    response = response.replace("```","").replace("markdown", "")
    update_display(Markdown(response), display_id=display_handle.display_id)


# OLLAMA RESPONSE WITH STREAMING
response = ollama.chat(
        model = MODEL_LLAMA,
        messages = [
            {"role":"system", "content":system_prompt},
            {"role":"user", "content":user_prompt}
        ]
    )

print("""## OLLAMA RESPONSE""")
display(Markdown(response['message']['content']))
