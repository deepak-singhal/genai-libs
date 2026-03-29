################################### FILE-1
####### RAG AGENT

import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
from huggingface_hub import login
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import chromadb
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from litellm import completion
from tqdm.notebook import tqdm
from agents.evaluator import evaluate
from agents.items import Item

####### ENVIRONMENT

load_dotenv(override=True)
DB = "DATABASE/products_vectorstore"
API_KEY=os.environ["OPEN_ROUTER_API_KEY"]
MODEL="openai/gpt-oss-20b:free"
BASE_URL="https://openrouter.ai/api/v1"
openai = OpenAI(base_url=BASE_URL, api_key=API_KEY)


####### HUGGINGFACE DATASET PULL UP
# Log in to HuggingFace
# If you don't have a HuggingFace account, you can set one up for free at www.huggingface.co
# And then add the HF_TOKEN to your .env file as explained in the project README

hf_token = os.environ['HF_TOKEN']
login(token=hf_token, add_to_git_credential=False)

LITE_MODE = True

username = "ed-donner"
dataset = f"{username}/items_lite" if LITE_MODE else f"{username}/items_full"
train, val, test = Item.from_hub(dataset)
print(f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items")



####### Vector DB Store
client = chromadb.PersistentClient(path=DB)
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Pass in a list of texts, get back a numpy array of vectors

# vector = encoder.encode(["A proficient AI engineer who has almost reached the finale of AI Engineering Core Track!"])[0]
# print(vector.shape)
# vector


# Check if the collection exists; if not, create it

collection_name = "products"
existing_collection_names = [collection.name for collection in client.list_collections()]

if collection_name not in existing_collection_names:
    collection = client.create_collection(collection_name)
    for i in tqdm(range(0, len(train), 1000)):
        documents = [item.summary for item in train[i: i+1000]]
        vectors = encoder.encode(documents).astype(float).tolist()
        metadatas = [{"category": item.category, "price": item.price} for item in train[i: i+1000]]
        ids = [f"doc_{j}" for j in range(i, i+1000)]
        ids = ids[:len(documents)]
        collection.add(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)

collection = client.get_or_create_collection(collection_name)


def vector(item):
    return encoder.encode(item.summary)


def find_similars(item):
    vec = vector(item)
    results = collection.query(query_embeddings=vec.astype(float).tolist(), n_results=5)
    documents = results['documents'][0][:]
    prices = [m['price'] for m in results['metadatas'][0][:]]
    return documents, prices

# print(test[0])
# find_similars(test[0])



# We need to give some context to GPT-5.1 by selecting 5 products with similar descriptions

def make_context(similars, prices):
    message = "For context, here are some other items that might be similar to the item you need to estimate.\n\n"
    for similar, price in zip(similars, prices):
        message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
    return message

# documents, prices = find_similars(test[0])
# print(make_context(documents, prices))

def messages_for(item, similars, prices):
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}\n\n"
    message += make_context(similars, prices)
    return [{"role": "user", "content": message}]

documents, prices = find_similars(test[0])
print(messages_for(test[0], documents, prices)[0]['content'])


# The function for llm call
def call_rag(item):
    documents, prices = find_similars(item)
    response = openai.chat.completions.create(model=MODEL, messages=messages_for(item, documents, prices), reasoning_effort="low", seed=42)
    return response.choices[0].message.content


call_rag(test[0])

############################################################################################################################################
################################### FILE 2 RAG + NEURAL NW => ENSEMBLE AGENT
import modal
Pricer = modal.Cls.from_name("pricer-service", "Pricer")
pricer = Pricer()

def specialist(item):
    return pricer.price.remote(item.summary)


def get_price(reply):
    reply = reply.replace("$", "").replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", reply)
    return float(match.group()) if match else 0


################################### NEURAL NW AGENT

from agents.deep_neural_network import DeepNeuralNetworkInference

runner = DeepNeuralNetworkInference()
runner.setup()
runner.load("deep_neural_network.pth")

def deep_neural_network(item):
    return runner.inference(item.summary)


# PUT WEIGHTS WHILE COMBINING THESE
def ensemble(item):
    price1 = get_price(call_rag(item))      # RAG MODEL
    price2 = specialist(item)               # FINE TUNED MODEL
    price3 = deep_neural_network(item)      # DEEP NEURAL NETWORK MODEL
    return price1 * 0.8 + price2 * 0.1 + price3 * 0.1

ensemble(test[0])

############################################################################################################################################
################################### FILE 3 
### ALL AGENTS

import logging
from agents.frontier_agent import FrontierAgent
from agents.neural_network_agent import NeuralNetworkAgent
from agents.ensemble_agent import EnsembleAgent

root = logging.getLogger()
root.setLevel(logging.INFO)

### RAG MODEL AGENT
rag_agent = FrontierAgent(collection)   # Collection is vectors defined above in upper cell
rag_agent.price("Quadcast HyperX condenser mic, connects via usb-c to your computer for crystal clear audio")
rag_agent.price("Shure MV7+ professional podcaster microphone with usb-c and XLR outputs")

# NEURAL NETWORK MODEL AGENT
neural_agent = NeuralNetworkAgent()
neural_agent.price("Shure MV7+ professional podcaster microphone with usb-c and XLR outputs")

# ENSEMBLE MODEL AGENT
ensemble_agent = EnsembleAgent(collection)
ensemble_agent.price("Shure MV7+ professional podcaster microphone with usb-c and XLR outputs")


