####### FILE-1
import os
import locale
import modal
from dotenv import load_dotenv
from hello import app, hello, hello_europe

load_dotenv(override=True)

with app.run():
    reply=hello.local()
print(reply)

with app.run():
    reply=hello.remote()
print(reply)

with app.run():
    reply=hello_europe.remote()
print(reply)




####### FILE-2

import modal
from modal import Image

app = modal.App("hello")
image = Image.debian_slim().pip_install("requests")

@app.function(image=image)
def hello() -> str:
    import requests

    response = requests.get("https://ipinfo.io/json")
    data = response.json()
    city, region, country = data["city"], data["region"], data["country"]
    return f"Hello from {city}, {region}, {country}!!"


@app.function(image=image, region="eu")
def hello_europe() -> str:
    import requests

    response = requests.get("https://ipinfo.io/json")
    data = response.json()
    city, region, country = data["city"], data["region"], data["country"]
    return f"Hello from {city}, {region}, {country}!!"
