# IMPORTS
import os
import openai
import gradio as gr
from dotenv import load_dotenv
import json

# ENVIRONMENT SETUP
MODEL_GPT="gpt-4"
load_dotenv(override=True)
openai.api_type=os.getenv('OPENAI_API_TYPE')
openai.azure_endpoint=os.getenv('OPENAI_API_BASE')
openai.api_version=os.getenv('OPENAI_API_VERSION')
openai.api_key=os.getenv('OPENAI_API_KEY')


# TOOLS : CUSTOM LOGIC FOR AI RESPONSES
ticket_prices={"delhi":"$1000", "toronto":"$50", "berlin":"$600", "cancun":"$100"}

def get_ticket_price(destination_city):
    city=destination_city.lower()
    return ticket_prices.get(city, "UNKNOWN CITY")
    
# There's a particular dictionary structure that's required to describe our function:
price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}

# And this is included in a list of tools:
tools = [{"type": "function", "function": price_function}]

def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    print(tool_call)
    arguments = json.loads(tool_call.function.arguments)
    city = arguments.get('destination_city')
    price = get_ticket_price(city)
    response = {
        "role": "tool",
        "content": json.dumps({"destination_city": city,"price": price}),
        "tool_call_id": tool_call.id
    }
    return response, city


# BUSINESS
system_message = "You are a helpful assistant for an Airline called FlightAI. "
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."                   # To avoid halucinating responses

def chat(message, history):
    messages = [{"role": "system", "content": system_message}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})
    response = openai.chat.completions.create(model=MODEL_GPT, messages=messages, tools=tools)  # Consult tools here
    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        response, city = handle_tool_call(message)
        messages.append(message)                  # Appending LLM message which is asking to run TOOLS
        messages.append(response)                 # Appending TOOLS result 
        response = openai.chat.completions.create(model=MODEL_GPT, messages=messages) # Then calling LLM with concatenated message (Sys+User+LLMtools+TOOLS)
    return response.choices[0].message.content


gr.ChatInterface(fn=chat).launch()
