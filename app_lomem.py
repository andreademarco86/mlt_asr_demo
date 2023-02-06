import os
import streamlit as st
import torch
from transformers import pipeline
import datetime
import gdown
from audio_recorder_streamlit import audio_recorder

st.set_page_config(
    page_title="MASRI Maltese ASR",
    page_icon="musical_note",
    layout="wide",
    initial_sidebar_state="auto",
)

audio_tags = {'comments': 'Converted using pydub!'}
upload_path = "uploads/"
download_path = "downloads/"
transcript_path = "transcripts/"
loaded = False

# check whether to run on CUDA, MPS, or CPU at beginning of the script
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    # Otherwise run on CPU by default
    device = torch.device("cpu")
PATH_TO_MODEL = './maltese_asr_model'  # can be local or online (huggingface) path

# Initialization of model
if 'asr_model' not in st.session_state:
    if not os.path.isfile(os.path.join(PATH_TO_MODEL,'pytorch_model.bin')):
        url = 'https://drive.google.com/file/d/1JvcgGmUVE29G4rTaadl-q_BZ1FI4GknP/view?usp=share_link'
        output_path = os.path.join(PATH_TO_MODEL, 'pytorch_model.bin')
        gdown.download(url, output_path, quiet=False, fuzzy=True)
    st.session_state['asr_model'] = pipeline("automatic-speech-recognition", model=PATH_TO_MODEL, device=device)
    # st.session_state['asr_model'] = pipeline("automatic-speech-recognition", model="MLRS/wav2vec2-xls-r-2b-mt-50", device=device, use_auth_token="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")
pipe = st.session_state['asr_model']


# @st.cache(persist=True,allow_output_mutation=False,show_spinner=True,suppress_st_warning=True)
def process_audio(audio_bytes,filename):
    with open(filename, "wb") as f:
        f.write(audio_bytes)
    print(filename)
    result = pipe(filename, chunk_length_s=10, stride_length_s=(4, 2), return_timestamps='word')
    os.remove(filename)
    return result["text"]


# @st.cache(persist=True,allow_output_mutation=False,show_spinner=True,suppress_st_warning=True)
def save_transcript(transcript_data, txt_file):
    with open(os.path.join(transcript_path, txt_file),"w") as f:
        f.write(transcript_data)


def reset():
    if 'output_file' in st.session_state:
        del st.session_state['output_file']

    if 'file_saved' in st.session_state:
        del st.session_state['file_saved']

    if 'filename' in st.session_state:
        del st.session_state['filename']

    if 'audio_loaded' in st.session_state:
        del st.session_state['audio_loaded']


st.title("🗣 Speech to Text for Maltese - The MASRI Engine")
st.info('Click icon to record and wait for colour to change. Recording will stop automatically after 2 seconds of silence.')

audio_bytes = audio_recorder(pause_threshold=2.0,sample_rate=16000,text="",recording_color="#e8b62c",neutral_color="#6aa36f",icon_size="6x")
if audio_bytes:
    # Initialization
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Feel free to play your recorded audio 🎼")
        st.audio(audio_bytes, format="audio/wav")
        if 'filename' not in st.session_state:
            st.session_state['filename'] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.wav'

    if st.button("Generate Transcript"):
        with st.spinner(f"Generating Transcript... 💫"):
            print(st.session_state['filename'])
            file_path = os.path.join(upload_path, st.session_state['filename'])
            transcript = process_audio(audio_bytes=audio_bytes,filename=os.path.abspath(file_path))
            st.markdown("Transcript: " + transcript)
            del st.session_state['filename']

else:
    reset()
    st.warning('⚠ Please record your audio 😯')

st.markdown("<br><hr><center>For feedback and queries, contact <a href='mailto:andrea.demarco@um.edu.mt?subject=ASR WebApp!&body=Please specify the issue you are facing with the app.'><strong>Dr Andrea DeMarco</strong></a>. Project site: [UMSpeech](https://www.um.edu.mt/projects/masri/) </center><hr>", unsafe_allow_html=True)


