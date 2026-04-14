# The imports
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, trace, set_default_openai_client, set_tracing_disabled


load_dotenv(override=True)
API_KEY=os.environ["OPENROUTER_API_KEY"]
MODEL="openai/gpt-oss-20b:free"
BASE_URL="https://openrouter.ai/api/v1"
openai = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
set_default_openai_client(openai)           # SET OPENROUTER CLIENT AS DEFAULT
set_tracing_disabled(True)                  # SET OBSERVABILITY TO OPENAI AS FALSE AS KEY IS NOT OPENAI ONE

agent = Agent(name="Jokester", instructions="You are a joke teller", model=MODEL)       # SYSTEM PROMPT

# agent
result = await Runner.run(agent, "Tell a joke about Vancouver Canada")                       # USER PROMPT
print(result.final_output)
