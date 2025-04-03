import os
import requests
from IPython.display import Markdown, display, update_display
import openai
from google.colab import drive
from huggingface_hub import login
from google.colab import userdata
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch

# CONSTANTS
AUDIO_MODEL = "openai/whisper-medium"
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# AUDIO FILE
# drive.mount("/content/drive")
audio_filename = "/content/sample_data/meeting.flac"

# LOGIN HUGGING FACE
hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

# GENERATE TRANSCRIPT USING WHISPER OPEN SOURCE MODEL
audio_file = open(audio_filename, "rb")
speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(AUDIO_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True).to('cuda')
processor = AutoProcessor.from_pretrained(AUDIO_MODEL)
pipe = pipeline(
    "automatic-speech-recognition",
    model=speech_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16,
    device='cuda',
)
result = pipe(audio_filename, return_timestamps=True)
transcription = result["text"]
print(transcription)

# PROMPTS
system_message = "You are an assistant that produces minutes of meetings from transcripts, with summary, key discussion points, takeaways and action items with owners, in markdown."
user_prompt = f"Below is an extract transcript of a Denver council meeting. Please write minutes in markdown, including a summary with attendees, location and date; discussion points; takeaways; and action items with owners.\n{transcription}"

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
  ]

# QUANTIZATION
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# GENERATE OUTPUT
tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto", quantization_config=quant_config)
outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)

response = tokenizer.decode(outputs[0])
display(Markdown(response))
