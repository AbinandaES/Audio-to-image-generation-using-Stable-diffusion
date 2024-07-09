import streamlit as st
import speech_recognition as sr
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
import base64
from PIL import Image
import os
from pydub import AudioSegment

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Initialize Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
sd_pipeline = StableDiffusionPipeline.from_pretrained(model_id)
sd_pipeline.to(device)

# Function to convert audio to text
def audio_to_text(audio_path):
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Request error from Google Speech Recognition service; {e}"

# Function to generate image from text prompt
def generate_image(prompt):
    image = sd_pipeline(prompt, guidance_scale=7.5).images[0]
    return image

# Function to save image to file
def save_image(image, path):
    image.save(path)

# Streamlit UI
st.title("InteriorGenius")

st.header("Audio to Image Generation")
uploaded_audio = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

record_audio_button = st.button("Record Audio")

# Initialize audio_path variable
audio_path = None

# Handle audio recording
if record_audio_button:
    st.write("Recording...")
    with st.spinner("Recording audio..."):
        # Use pydub to record audio (requires arecord or similar tool)
        os.system("arecord -D plughw:1,0 -d 5 -f cd audio.wav")  # Adjust the command according to your system
        audio_path = "audio.wav"
elif uploaded_audio is not None:
    audio_path = uploaded_audio

# Convert audio to text if audio_path is set
if audio_path:
    text = audio_to_text(audio_path)
    st.write("Transcribed Text:", text)
    text_prompt = text
else:
    text_prompt = st.text_input("Or enter text prompt:")

# Generate image from text prompt
if st.button("Generate Image"):
    if text_prompt:
        with st.spinner("Generating image please wait..."):
            image = generate_image(text_prompt)
            st.image(image, caption="Generated Image")
            
            # Allow downloading the image
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            href = f'<a href="data:image/png;base64,{img_str}" download="generated_image.png">Download Image</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Allow uploading the image
            save_path = st.text_input("Enter the file path to save the image (e.g., './generated_image.png'):")
            if st.button("Save Image"):
                if save_path:
                    save_image(image, save_path)
                    st.write(f"Image saved to {save_path}")
    else:
        st.error("Please provide a text prompt.")
 
 