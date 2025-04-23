# imports
import os
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, Dataset, DatasetDict
import matplotlib.pyplot as plt
from items import Item

# environment
load_dotenv(override=True)
HF_TOKEN = os.getenv('HF_TOKEN')

# Log in to HuggingFace
login(HF_TOKEN, add_to_git_credential=True)

################################# Load the Dataset #################################
%matplotlib inline
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_Appliances", split="full", trust_remote_code=True)
# print(f"Number of Appliances: {len(dataset):,}")

################################# Analyze the Data #################################

# Investigate a particular datapoint
# dataset[2]

# Check the data with price=0
# prices = 0
# for datapoint in dataset:
#     try:
#         price = float(datapoint["price"])
#         if price > 0:
#             prices += 1
#     except ValueError as e:
#         pass

# print(f"There are {prices:,} with prices which is {prices/len(dataset)*100:,.1f}%")

# For those with prices, gather the price and the length(characters lengths in title+description+features+details)
prices = []
lengths = []
for datapoint in dataset:
    try:
        price = float(datapoint["price"])
        if price > 0:
            prices.append(price)
            contents = datapoint["title"] + str(datapoint["description"]) + str(datapoint["features"]) + str(datapoint["details"])
            lengths.append(len(contents))
    except ValueError as e:
        pass

# Plot the distribution of lengths vs ApplianceCounts
plt.figure(figsize=(15, 6))
plt.title(f"Lengths: Avg {sum(lengths)/len(lengths):,.0f} and highest {max(lengths):,}\n")
plt.xlabel('Length (chars)')
plt.ylabel('Count')
plt.hist(lengths, rwidth=0.7, color="lightblue", bins=range(0, 6000, 100))
plt.show()

# Plot the distribution of prices vs ApplianceCounts
plt.figure(figsize=(15, 6))
plt.title(f"Prices: Avg {sum(prices)/len(prices):,.2f} and highest {max(prices):,}\n")
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.hist(prices, rwidth=0.7, color="orange", bins=range(0, 1000, 10))
plt.show()


################################## Curate the Data #################################
items = []
for datapoint in dataset:
    try:
        price = float(datapoint["price"])
        if price > 0:
            item = Item(datapoint, price)
            if item.include:
                items.append(item)
    except ValueError as e:
        pass

print(f"There are {len(items):,} items")


# Look at the specific item
# items[10]

# Investigate the prompt that will be used during training - the model learns to complete this
# print(items[0].prompt)

# Investigate the prompt that will be used during testing - the model has to complete this
# print(items[100].test_prompt())

# Plot the distribution of token counts
tokens = [item.token_count for item in items]
plt.figure(figsize=(15, 6))
plt.title(f"Token counts: Avg {sum(tokens)/len(tokens):,.1f} and highest {max(tokens):,}\n")
plt.xlabel('Length (tokens)')
plt.ylabel('Count')
plt.hist(tokens, rwidth=0.7, color="green", bins=range(0, 300, 10))
plt.show()

# Plot the distribution of prices
prices = [item.price for item in items]
plt.figure(figsize=(15, 6))
plt.title(f"Prices: Avg {sum(prices)/len(prices):,.1f} and highest {max(prices):,}\n")
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.hist(prices, rwidth=0.7, color="purple", bins=range(0, 300, 10))
plt.show()
