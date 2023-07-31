from dotenv import load_dotenv,find_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI

from transformers import pipeline
import requests
import os
# Continue with the rest of your code
import streamlit as st

load_dotenv(find_dotenv())
hugging_face_key = os.getenv("HUGGING_FACE_KEY")
#image2text
def image2text(img_path):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

    text = image_to_text(img_path)
    print(text)
    return text[0]["generated_text"]


#llm
def generate_story(scenario):
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
    print(story)
    return story


#text to speech
def text_to_speech(text):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {hugging_face_key}"}
    payloads = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as f:
        f.write(response.content)

#scenario = image2text("myself.jpg")
#short_story=generate_story(scenario)
#text_to_speech(short_story)
# Streamlit app
st.title('Image to Short Story')
st.write('Upload an image, and I will create a short story based on it!')

uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'png'])
if uploaded_image:
    # Save the uploaded image temporarily
    img_path = 'uploaded_image.jpg'
    with open(img_path, 'wb') as f:
        f.write(uploaded_image.read())

    # Use your existing functions to process the image
    scenario = image2text(img_path)
    short_story = generate_story(scenario)
    text_to_speech(short_story)

    # Display the resulting story
    st.subheader('Generated Short Story:')
    st.write(short_story)
    st.audio('audio.flac')

    # Optionally, delete the temporary files
    os.remove(img_path)
    os.remove('audio.flac')