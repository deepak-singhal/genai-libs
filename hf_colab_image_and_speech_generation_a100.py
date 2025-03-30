#####For HuggingFace & GoogleColab#####
from huggingface_hub import login
from google.colab import userdata
#########For Image Generation##########
import torch
from diffusers import FluxPipeline
from IPython.display import display
from PIL import Image
#####For Text to Speech Generation#####
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch
from IPython.display import Audio


# Login HF with Token which is fetched from Colab Keys 
hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

# Image Generation : only work on a powerful GPU box like an A100
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
generator = torch.Generator(device="cuda").manual_seed(0)
prompt = "A futuristic class full of students learning AI coding in the surreal style of Salvador Dali"

image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=generator
).images[0]

image.save("surreal.png")
display(image)


# Speech Generation : only work on a powerful GPU box like an A100
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device='cuda')

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = synthesiser("Hi to an artificial intelligence engineer on the way to mastery!", forward_params={"speaker_embeddings": speaker_embedding})

sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])

