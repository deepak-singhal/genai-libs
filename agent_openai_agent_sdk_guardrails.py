import os
import certifi
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, trace, function_tool, input_guardrail, output_guardrail, GuardrailFunctionOutput, set_default_openai_client, set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
from typing import Dict
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content
import asyncio
from pydantic import BaseModel


os.environ['SSL_CERT_FILE'] = certifi.where()


load_dotenv(override=True)
API_KEY=os.environ["OPENROUTER_API_KEY"]
MODEL="openai/gpt-4o-mini"
BASE_URL="https://openrouter.ai/api/v1"
openai = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
set_default_openai_client(openai)           # SET OPENROUTER CLIENT AS DEFAULT
set_tracing_disabled(True)                  # SET OBSERVABILITY TO OPENAI AS FALSE AS KEY IS NOT OPENAI ONE


# DEFINE AGENTS
instructions1 = "You are a sales agent working for ComplAI, \
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
You write professional, serious cold emails."

instructions2 = "You are a humorous, engaging sales agent working for ComplAI, \
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
You write witty, engaging cold emails that are likely to get a response."

instructions3 = "You are a busy sales agent working for ComplAI, \
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
You write concise, to the point cold emails."

subject_instructions = "You can write a subject for a cold sales email. \
You are given a message and you need to write a subject for an email that is likely to get a response."

html_instructions = "You can convert a text email body to an HTML email body. \
You are given a text email body which might have some markdown \
and you need to convert it to an HTML email body with simple, clear, compelling layout and design."

sales_agent1 = Agent(name="Professional Sales Agent", instructions=instructions1, model=MODEL)
sales_agent2 = Agent(name="Engaging Sales Agent", instructions=instructions2, model=MODEL)
sales_agent3 = Agent(name="Busy Sales Agent", instructions=instructions3, model=MODEL)
subject_writer_agent = Agent(name="Email subject writer", instructions=subject_instructions, model=MODEL)
html_converter_agent = Agent(name="HTML email body converter", instructions=html_instructions, model=MODEL)



# DEFINE TOOLS
tool1 = sales_agent1.as_tool(tool_name="sales_agent1", tool_description="Write a cold sales email")
tool2 = sales_agent2.as_tool(tool_name="sales_agent2", tool_description="Write a cold sales email")
tool3 = sales_agent3.as_tool(tool_name="sales_agent3", tool_description="Write a cold sales email")
subject_tool = subject_writer_agent.as_tool(tool_name="subject_writer", tool_description="Write a subject for a cold sales email")
html_tool = html_converter_agent.as_tool(tool_name="html_converter", tool_description="Convert a text email body to an HTML email body")

@function_tool
def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
    """ Send out an email with the given subject and HTML body to all sales prospects """
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
    from_email = Email("duskydeeps@gmail.com")  # Change to your verified sender
    to_email = To("singhal03.deepak@gmail.com")  # Change to your recipient
    content = Content("text/html", html_body)
    mail = Mail(from_email, to_email, subject, content).get()
    sg.client.mail.send.post(request_body=mail)
    print(f"Best email sent : {html_body}")
    return {"status": "success"}

email_draft_tools = [tool1, tool2, tool3]
handoff_tools = [subject_tool, html_tool, send_html_email]



# DEFINE HAND-OFFS = subject_tool + html_tool + send_html_email_tool/ WITH HAND OFF DESCRIPTION
email_format_instructions ="You are an email formatter and sender. You receive the body of an email to be sent. \
You first use the subject_writer tool to write a subject for the email, then use the html_converter tool to convert the body to HTML. \
Finally, you use the send_html_email tool to send the email with the subject and HTML body."
emailer_agent = Agent(name="Email Manager", instructions=email_format_instructions, tools=handoff_tools, model=MODEL, handoff_description="Convert an email to HTML & send")

handoffs = [emailer_agent]


############################ GUARDRAILS IMPLEMENTATIONS ########################################################

# Pydantic class for structured output
class NameCheckOutput(BaseModel):
    is_name_in_message: bool
    name: str
    rule: str
    solution: str


guardrail_agent = Agent( 
    name="Name check",
    instructions="Check if the user is including someone's personal name in what they want you to do.",
    output_type=NameCheckOutput,    # Providing structure output class name to map the fields
    model="gpt-4o-mini"
)

@input_guardrail
async def guardrail_against_name(ctx, agent, message):
    result = await Runner.run(guardrail_agent, message, context=ctx.context)
    print(f"INPUT GUARDRAIL : {result.final_output}")
    is_name_in_message = result.final_output.is_name_in_message
    return GuardrailFunctionOutput(output_info={"found_name": result.final_output},tripwire_triggered=is_name_in_message)

################################################################################################################


############################ DEFINE MAIN AGENT ########################################################
sales_manager_instructions = """
You are a Sales Manager at ComplAI. Your goal is to find the single best cold sales email using the sales_agent tools.
 
Follow these steps carefully:
1. Generate Drafts: Use all three sales_agent tools to generate three different email drafts. Do not proceed until all three drafts are ready.
 
2. Evaluate and Select: Review the drafts and choose the single best email using your judgment of which one is most effective.
You can use the tools multiple times if you're not satisfied with the results from the first try.
 
3. Handoff for Sending: Pass ONLY the winning email draft to the 'Email Manager' agent. The Email Manager will take care of formatting and sending.
 
Crucial Rules:
- You must use the sales agent tools to generate the drafts — do not write them yourself.
- You must hand off exactly ONE email to the Email Manager — never more than one.
"""


sales_manager = Agent(
    name="Sales Manager",
    instructions=sales_manager_instructions,
    tools=email_draft_tools,
    handoffs=handoffs,
    input_guardrails=[guardrail_against_name],
    model=MODEL)

message = "Send out a cold sales email addressed to Dear CEO."

result = await Runner.run(sales_manager, message)
##################################################################################################
