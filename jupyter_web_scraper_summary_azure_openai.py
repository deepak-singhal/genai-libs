# imports

import os
import requests
import openai
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display

# Load environment variables in a file called .env

load_dotenv(override=True)
openai.api_type=os.getenv('OPENAI_API_TYPE')
openai.azure_endpoint=os.getenv('OPENAI_API_BASE')
openai.api_version=os.getenv('OPENAI_API_VERSION')
openai.api_key=os.getenv('OPENAI_API_KEY')

# Check the key
if not openai.api_key:
    print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
elif openai.api_key.strip() != openai.api_key:
    print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")

# A class to represent a Webpage
# If you're not familiar with Classes, check out the "Intermediate Python" notebook
# Some websites need you to use proper headers when fetching them:
headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:
    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
        self.url = url
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

ed = Website("https://edwarddonner.com")
# print(ed.title)
# print(ed.text)

#A system prompt -- that tells them what task they are performing and what tone they should use
#A user prompt   -- the conversation starter that they should reply to

system_prompt = "You are an assistant that analyzes the contents of a website \
and provides a short summary, ignoring text that might be navigation related. \
Respond in markdown."

def user_prompt_for(website):
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += "\nThe contents of this website is as follows; \
                       Please provide a short summary of this website in markdown. \
                       If it includes news or announcements, then summarize these too.\n\n"
    user_prompt += website.text
    return user_prompt


# Combining Messages
def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)}
    ]


def summarize(url):
    website = Website(url)
    response = openai.chat.completions.create(
        model = "gpt-4",
        messages = messages_for(website)
    )
    return response.choices[0].message.content

summary = summarize("https://cnn.com/")
display(Markdown(summary))
