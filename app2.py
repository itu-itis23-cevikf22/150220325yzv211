import streamlit as st
from transformers import pipeline
import os


# Hi Teacher, I had to add this "ffmpeg" path, I could not find another way.
os.environ["PATH"] += os.pathsep + r"C:\Users\Faruk\Downloads\ffmpeg-master-latest-win64-gpl-shared 2\ffmpeg-master-latest-win64-gpl-shared\bin"

# ------------------------------
# Whisper Model
# ------------------------------
def load_whisper_model():
    """
    Loads the Whisper Model.
    """
    model = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    return model

# ------------------------------
# NER Model 
# ------------------------------
def load_ner_model():
    """
    Loads the NER Model.
    """
    ner_model = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return ner_model

# ------------------------------
# Transcribe audio to text
# ------------------------------
def transcribe_audio(uploaded_file, model):
    """
    Transcribes the audio file to a text using Whisper Model.
    """
    audio = uploaded_file.read()
    transcription = model(audio, return_timestamps=True)
    return transcription["text"]

# ------------------------------
# Extract entities
# ------------------------------
def extract_entities(text, ner_model):
    """
    Extracts persons (PERs), organizations (ORGs) and locations (LOCs) from the text.
    """
    entities = ner_model(text)
    
    # Extract Organizations (ORGs)
    orgs = set([entity['word'] for entity in entities if entity['entity_group'] == 'ORG'])
    
    # Extract Locations (LOCs) 
    locs = set([entity['word'] for entity in entities if entity['entity_group'] == 'LOC'])
    
    # Extract persons (PERs) 
    pers = set([entity['word'] for entity in entities if entity['entity_group'] == 'PER'])
    
    return orgs, locs, pers
# ------------------------------
# Streamlit Application
# ------------------------------
def main():
    st.title("Extracting Text and Entities From Audio")


    STUDENT_NAME = "Faruk"
    STUDENT_ID = "150220325"
    st.write(f"**Student ID: {STUDENT_ID} - {STUDENT_NAME}**")

   
    uploaded_file = st.file_uploader("Upload the audio file", type=["wav"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")  # Audio File
        
        # Whisper model
        whisper_model = load_whisper_model()
        
        # NER model
        ner_model = load_ner_model()
        
        
        transcribed_text = transcribe_audio(uploaded_file, whisper_model)
        st.write("# Transcription")
        st.text(transcribed_text)

        st.write("# Extracted Entities:")
        orgs, locs, pers = extract_entities(transcribed_text, ner_model)
        
        # Group and display the entities
        st.write("## Organizations (ORGs):")
        for entity in orgs:
            st.write(f"• {entity}")
        st.write("## Locations (LOCs):")
        for entity in locs:
            st.write(f"• {entity}")
        st.write("## Persons (PERs):")
        for entity in pers:
            st.write(f"• {entity}")

if __name__ == "__main__":
    main()


