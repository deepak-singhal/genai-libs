import os
from dotenv import load_dotenv
import anthropic
from IPython.display import Markdown, display, update_display


load_dotenv(override=True)
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
claude = anthropic.Anthropic()

system_message = "You are an assistant that is great at telling jokes"
user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"


# Normal Message Delivery
message = claude.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=200,
    temperature=0.7,
    system=system_message,
    messages=[
        {"role": "user", "content": user_prompt},
    ],
)
print(message.content[0].text)


# Stream Message Delivery
result = claude.messages.stream(
    model="claude-3-5-sonnet-latest",
    max_tokens=200,
    temperature=0.7,
    system=system_message,
    messages=[
        {"role": "user", "content": user_prompt},
    ],
)
with result as stream:
    for text in stream.text_stream:
            print(text, end="", flush=True)
