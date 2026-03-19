# Fine Tuning a Frontier Model : OPENAI GPT
# Supervized Fine Tuning (SFT)

import os
import re
import json
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
from pricer.items  import Item
from pricer.evaluator import evaluate

LITE_MODE = True

load_dotenv(override=True)
HF_TOKEN = os.environ['HF_TOKEN']
API_KEY=os.environ["OPEN_ROUTER_API_KEY"]
# MODEL="openai/gpt-oss-20b:free"
MODEL="openai/gpt-4.1-nano"
BASE_URL="https://openrouter.ai/api/v1"

login(HF_TOKEN, add_to_git_credential=True)
username = "dsinghal89"
dataset = f"{username}/items_lite" if LITE_MODE else f"{username}/items_full"
train, val, test = Item.from_hub(dataset)
print(f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items")

openai=OpenAI(base_url=BASE_URL, api_key=API_KEY)

# OpenAI recommends fine-tuning with populations of 50-100 examples
# But as our examples are very small, I'm suggesting we go with 100 examples (and 1 epoch)

fine_tune_train = train[:100]
fine_tune_validation = val[:50]


# Step 1 : Prepare our data for fine-tuning in JSONL (JSON Lines) format and upload to OpenAI
def messages_for(item):
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}"
    return [
        {"role": "user", "content": message},
        {"role": "assistant", "content": f"${item.price:.2f}"}
    ]

def make_jsonl(items):
    result = ""
    for item in items:
        messages = messages_for(item)
        messages_str = json.dumps(messages)
        result += '{"messages": ' + messages_str +'}\n'
    return result.strip()


def write_jsonl(items, filename):
    with open(filename, "w") as f:
        jsonl = make_jsonl(items)
        f.write(jsonl)

write_jsonl(fine_tune_train, "jsonl/fine_tune_train.jsonl")
write_jsonl(fine_tune_validation, "jsonl/fine_tune_validation.jsonl")

# Upload these files to OpenAI for FineTuning [this is similar to the batch operation we did in GROQ]
# The file status can be checked at : https://platform.openai.com/storage/files/
with open("jsonl/fine_tune_train.jsonl", "rb") as f:
    train_file = openai.files.create(file=f, purpose="fine-tune")

with open("jsonl/fine_tune_validation.jsonl", "rb") as f:
    validation_file = openai.files.create(file=f, purpose="fine-tune")



# Step 2 : And now time to Fine-tune! 
# Check the status of training : https://platform.openai.com/finetune
openai.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=validation_file.id,
    model=MODEL,
    seed=42,
    hyperparameters={"n_epochs": 1, "batch_size": 1},
    suffix="pricer"
)

job_id = openai.fine_tuning.jobs.list(limit=1).data[0].id
openai.fine_tuning.jobs.retrieve(job_id)
openai.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10).data



# Step 3 : Test our fine tuned model
fine_tuned_model_name = openai.fine_tuning.jobs.retrieve(job_id).fine_tuned_model

def test_messages_for(item):
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}"
    return [
        {"role": "user", "content": message},
    ]

# test_messages_for(test[0])

def gpt_4__1_nano_fine_tuned(item):
    response = openai.chat.completions.create(
        model=fine_tuned_model_name,
        messages=test_messages_for(item),
        max_tokens=7
    )
    return response.choices[0].message.content


print("Actual Price : ", test[0].price)
print("Gussed Price by fine tuned Model : ", gpt_4__1_nano_fine_tuned(test[0]))

evaluate(gpt_4__1_nano_fine_tuned, test)
