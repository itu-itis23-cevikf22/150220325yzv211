import streamlit as st
from transformers import pipeline
import os


# FFmpeg'in bulunduğu dizini PATH'e ekle
os.environ["PATH"] += os.pathsep + r"C:\Users\Faruk\Downloads\ffmpeg-master-latest-win64-gpl-shared 2\ffmpeg-master-latest-win64-gpl-shared\bin"

# ------------------------------
# Whisper Modelini Yükle
# ------------------------------
def load_whisper_model():
    """
    Whisper modelini yükler ve geri döner.
    """
    model = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    return model

# ------------------------------
# NER Modelini Yükle 
# ------------------------------
def load_ner_model():
    """
    NER modelini yükler ve geri döner.
    """
    ner_model = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return ner_model

# ------------------------------
# Ses Dosyasını Metne Dönüştür
# ------------------------------
def transcribe_audio(uploaded_file, model):
    """
    Yüklenen ses dosyasını Whisper modelini kullanarak metne dönüştürür.
    """
    audio = uploaded_file.read()
    transcription = model(audio, return_timestamps=True)
    return transcription["text"]

# ------------------------------
# Varlık Çıkartma
# ------------------------------
def extract_entities(text, ner_model):
    """
    Metinden kişileri (PERs), organizasyonları (ORGs) ve lokasyonları (LOCs) çıkarır.
    """
    entities = ner_model(text)
    
    # Organizasyonları (ORGs) çıkar
    orgs = set([entity['word'] for entity in entities if entity['entity_group'] == 'ORG'])
    
    # Lokasyonları (LOCs) çıkar
    locs = set([entity['word'] for entity in entities if entity['entity_group'] == 'LOC'])
    
    # Kişileri (PERs) çıkar
    pers = set([entity['word'] for entity in entities if entity['entity_group'] == 'PER'])
    
    return orgs, locs, pers
# ------------------------------
# Streamlit Uygulaması
# ------------------------------
def main():
    st.title("Extracting Text and Entities From Audio")


    STUDENT_NAME = "Faruk"
    STUDENT_ID = "150220325"
    st.write(f"**Student ID: {STUDENT_ID} - {STUDENT_NAME}**")

    # Dosya Yükleyici
    uploaded_file = st.file_uploader("Upload the audio file", type=["wav"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")  # Ses dosyası
        
        # Whisper modeli
        whisper_model = load_whisper_model()
        
        # NER modeli
        ner_model = load_ner_model()
        
        # Ses kaydını metne dönüştür
        transcribed_text = transcribe_audio(uploaded_file, whisper_model)
        st.write("# Transcription")
        st.text(transcribed_text)

        # Varlıkları çıkar
        st.write("# Extracted Entities:")
        orgs, locs, pers = extract_entities(transcribed_text, ner_model)
        
        # Varlıkları gruplandır ve göster
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


