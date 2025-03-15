# 1. imports

import ollama
import requests
from bs4 import BeautifulSoup
from IPython.display import Markdown, display

# 2. Load environment variables in a file called .env

MODEL = "llama3.2"

# 3. Check the key


# 4. Prompts
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


# 5. Combining Messages
def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)}
    ]


# 6. Business Logic
class Website:
    def __init__(self, url):
        self.url = url
        headers = {
         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)


def summarize(url):
    website = Website(url)
    response = ollama.chat(model=MODEL, messages=messages_for(website))
    return response['message']['content']


summary = summarize("https://cnn.com/")
display(Markdown(summary))
