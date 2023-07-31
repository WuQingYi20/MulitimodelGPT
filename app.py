import streamlit as st
from dotenv import load_dotenv,find_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
import requests
import os

# Loading the environment variables
load_dotenv(find_dotenv())
hugging_face_key = os.getenv("HUGGING_FACE_KEY")
if hugging_face_key:
    st.write("Successfully loaded Hugging Face key.")
else:
    st.write("Failed to load Hugging Face key.")

def image2text(filename):
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    headers = {"Authorization": f"Bearer {hugging_face_key}"}

    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()[0]["generated_text"]


def generate_story(scenario):
    try:
        template = """
        you are a story teller.
        you can generate a short story based on a simple narrative, the story should be no more than 20 words;
        CONTEXT:{scenario}
        STORY:
        """
        prompt = PromptTemplate(template = template, input_variables=["scenario"])
        story_llm = LLMChain(llm = OpenAI(
            model_name ="gpt-3.5-turbo",temperature= 1), prompt=prompt, verbose =True)
        story = story_llm.predict(scenario = scenario)
        st.write(f"Generated story: {story}")
        return story
    except Exception as e:
        st.write(f"Error in generate_story: {str(e)}")

def text_to_speech(text):
    try:
        API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
        headers = {"Authorization": f"Bearer {hugging_face_key}"}
        payloads = {"inputs": text}
        response = requests.post(API_URL, headers=headers, json=payloads)
        with open('audio.flac', 'wb') as f:
            f.write(response.content)
    except Exception as e:
        st.write(f"Error in text_to_speech: {str(e)}")

# Streamlit app
st.title('Image to Short Story')
st.write('Upload an image, and I will create a short story based on it!')

uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'png'])
if uploaded_image:
    img_path = 'uploaded_image.jpg'
    with open(img_path, 'wb') as f:
        f.write(uploaded_image.read())
    st.write(f"Uploaded image saved at path: {img_path}")

    scenario = image2text(img_path)
    short_story = generate_story(scenario)
    text_to_speech(short_story)

    st.subheader('Generated Short Story:')
    st.write(short_story)
    st.audio('audio.flac')

    os.remove(img_path)
    st.write(f"Deleted temporary image file at path: {img_path}")
    os.remove('audio.flac')
    st.write("Deleted temporary audio file.")
