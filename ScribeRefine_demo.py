# Import libraries
import streamlit as st
import openai
from openai import OpenAI
import google.generativeai as genai
import assemblyai as aai
import base64
import tempfile

def img_to_base64(image_path):
    """Convert image to base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        return None
st.set_page_config(page_title="ScribeRefine", layout="centered")





PROMPT_TEMPLATE = """
You are a meticulous and highly skilled editor specializing in the post-processing of Automatic Speech Recognition (ASR) transcripts. 
Your sole function is to identify and correct errors in a given transcript chunk while preserving the original speaker's intent and meaning.

Follow these steps with precision:

Step 1: **Identify and Correct Errors**
   - Correct all spelling mistakes. Use '{language}' spelling conventions.
   - Rectify grammatical errors to ensure the text is coherent.
   - Insert appropriate punctuation (commas, periods, question marks, etc.) to reflect natural speech patterns.

Step 2: **Sanitize the Transcript**
   - Remove any and all formatting characters (e.g., \\n, \\r).
   - Erase any timestamps (e.g., [00:12:34.567]) and speaker labels (e.g., "SPEAKER_01:", "John Doe:").

Step 3: **Adhere to Strict Constraints**
   - **DO NOT** add new information.
   - **DO NOT** rephrase sentences or alter the core vocabulary. The original meaning must be strictly maintained.
   - **DO NOT** omit any part of the original content.
   - **DO NOT** include any introductory phrases, summaries, or commentary in your output.

{error_examples}

{context}

Now, apply these rules to the following transcript chunk.

### Transcript Chunk to Correct ###
"{transcript_chunk}"
### END ###

### Corrected Output ###
"""

PROMPT_TEMPLATE2 = """
The following text is a small chunk from a larger ASR transcript.
-You are an error-correction assistant for ASR (Automatic Speech Recognition) transcripts. Given a short transcript chunk, apply the following corrections:
1. Fix all spelling, grammar, and punctuation errors using '{language}' spelling.
2. Do not add, rephrase, or omit content. Preserve the original meaning strictly.
3. Remove all formatting characters (e.g., \\n, \\r).
4. Remove any timestamps and speaker identifiers.
5. Do not add any commentary or introductory phrases.
6. Do not use any word compression techniques like 'don't' or 'can't', use 'do not' and 'cannot' instead.

7. ---- Please make sure hat you correct on the basis of 
        - error examples, 
        - transcript chunk (the Main script is divided into multiple semantic chunks),
        - summary for the whole script that is provided,

--- Context for the video-transcript: 
            ---- Start of context ----
            "{context}"
            ---- End of context ----

--- Transcript Chunk to correct: 
            ---- Start  of Chunk----
            "{transcript_chunk}"
            ---- End of Chunk----



Corrected Output:
"""


ERROR_EXAMPLES = """**Examples of Successful Corrections:**

**Example 1:**
- **Transcript Chunk to Correct:**
  "so whats the plan you re still goin to the theatre later right cause i wasnt sure if we had to leave now"
- **Corrected Output:**
  "So, what's the plan? You are still going to the theatre later, right? Because I was not sure if we had to leave now."

**Example 2:**
- **Transcript Chunk to Correct:**
  "my favorite class is history but seth s is maths you know i dont think he likes his neighborhood very much"
- **Corrected Output:**
  "My favourite class is history, but Sally's is maths. You know, I do not think he likes his neighbourhood very much."

**Example 3:**
- **Transcript Chunk to Correct:**
  "we ve been there before i think anytime youe going its a good time but we didn t have the chance"
- **Corrected Output:**
  "We have been there before. I think any time you are going, it's a good time, but we did not have the chance."

**Example 4:**
- **Transcript Chunk to Correct:**
  "SPEAKER_02: [01:14:23.451] that character joe polnacek s behaviour isnt my favorite i wouldn t do that myself"
- **Corrected Output:**
  "That character Jo Polniaczek's behaviour is not my favourite. I would not do that myself."

**Example 5:**
- **Transcript Chunk to Correct:**
  "i m going to cover the main points um we dont wanna forget the colourful illustrations which are a big part of it"
- **Corrected Output:**
  "I am going to cover the main points. Um, we do not want to forget the colourful illustrations, which are a big part of it."
"""













#st.title("ScribeRefine")
st.markdown(
    f"<h1 style='color:#a12b46;'>ScribeRefine</h1>",
    unsafe_allow_html=True
)
st.markdown("Upload media file (WAV or MP3) → Transcribe with AssemblyAI → Improve transcript quality using GenAI!")

genai.configure(api_key= st.secrets["api_keys"]["GEMINI_API_KEY"])
openai.api_key = st.secrets["api_keys"]["OPENAI_API_KEY"]
aai.settings.api_key = st.secrets["api_keys"]["ASSEMBLYAI_API_KEY"]

if not openai.api_key:
    st.error("No API key set. Use secrets.toml or environment variable.")
    st.stop()

img_path = "logo.png"
img_base64 = img_to_base64(img_path)
if img_base64:
    st.sidebar.markdown(f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">', unsafe_allow_html=True)
st.sidebar.markdown(
    """
        <br>

    **ScribeRefine** is a powerful tool designed by ScribeWire to enhance the quality of audio transcriptions. 
    It allows you to:
    - Upload audio files (WAV or MP3).
    - Transcribe them using common ASR models.
    - Refine the transcription with context-aware Large Language Models.
    """ , unsafe_allow_html=True)

st.sidebar.header("Speech Recognition Model")
asr_mode = st.sidebar.radio("Choose between:", ["🎯 AssemblyAI Slam-1", "🔒 OpenAI Whisper Large V2"])


st.markdown(
    f"<h2 style='color:#2b5b85;'>Upload Media File</h2>",
    unsafe_allow_html=True
)
#st.header("Upload Media File")
audio_file = st.file_uploader("Upload a WAV or MP3 file for the content you want to transcribe!", type=["wav", "mp3"])


if audio_file and st.button("Transcribe Audio"):
    st.session_state.audio_path = audio_file

if st.session_state.get("audio_path") and not st.session_state.get("transcript_complete"):
    if asr_mode == "🔒 OpenAI Whisper Large V2":
        st.header("Whisper Transcription")
        status_placeholder = st.empty()
        status_placeholder.info("Transcribing with Whisper ...")
        try:
            openai_client = OpenAI(api_key=openai.api_key)
            with st.spinner("Waiting for OpenAI ..."):
                st.session_state.transcript_response = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=st.session_state.audio_path).text

                status_placeholder.success("Transcription completed ✅")      
                st.session_state.transcript_complete = True

        except Exception as e:
            status_placeholder.error("Transcription failed ❌")
            st.error(f"{e}")
            st.stop()

    else:
        st.header("AssemblyAI Transcription")
        status_placeholder = st.empty()
        status_placeholder.info("Transcribing with AssemblyAI ...")
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(st.session_state.audio_path.getvalue())
                st.session_state.audio_path = tmp_file.name
                transcriber = aai.Transcriber()
                with st.spinner("Waiting for AssemblyAI ..."):
                    st.session_state.transcript_response = transcriber.transcribe(st.session_state.audio_path, aai.TranscriptionConfig(speech_model="slam-1")).text

                    status_placeholder.success("Transcription completed ✅")      
                    st.session_state.transcript_complete = True

        except Exception as e:
            status_placeholder.error("Transcription failed ❌")
            st.error(f"{e}")
            st.stop()


st.sidebar.header("Error Correction LLM")
st.session_state.llm_mode = st.sidebar.radio("Choose between:", ["OpenAI GPT-4o mini", "Google Gemini 2.5 Flash"])

if st.session_state.get("transcript_complete"):

    st.write("📝 ASR Transcript 📝:")
    st.text(st.session_state.transcript_response)
    st.checkbox("I have additional context for this media file!", value=False, key="enable_context")

    if st.session_state.get("enable_context"):
        context_example = "This transcript features a discussion about the impact of climate change on local ecosystems. The speakers discuss various species and their adaptations to changing environments."
        st.session_state.context = st.text_area("Context", value=context_example, height=200)

    if st.button("Apply LLM"):  
        st.session_state.correction_ready = True

if st.session_state.get("correction_ready"):
    #st.header("Refining Transcript with LLM")
    st.markdown(
        f"<h2 style='color:#fb9d1d;'>Refining Transcript with LLM</h2>",
        unsafe_allow_html=True
    )
    
    
    if not st.session_state.get('context'):
        st.session_state.context = ""
    if st.session_state.llm_mode == "OpenAI GPT-4o mini":
        llm_model = "gpt-4o-mini"
        prompt = PROMPT_TEMPLATE.format(transcript_chunk=st.session_state.transcript_response,
                                        context=st.session_state.context,
                                        language='en-CA', 
                                        error_examples=ERROR_EXAMPLES)
        with st.spinner(f"Correcting transcript using {llm_model} ..."):
            try:
                client = OpenAI(api_key=openai.api_key)
                completion = client.chat.completions.create(
                    model=llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    **{"temperature": 0.1}
                )
                corrected = completion.choices[0].message.content
                st.text_area("✅ Corrected Output", corrected)

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        llm_model = "gemini-2.5-flash"
        prompt = PROMPT_TEMPLATE2.format(transcript_chunk=st.session_state.transcript_response,
                                        context=st.session_state.context,
                                        language='en-CA', 
                                        error_examples=ERROR_EXAMPLES)
        
        with st.spinner(f"Correcting transcript using {llm_model} ..."):
            try:
                client = genai.GenerativeModel(llm_model)
                completion = client.generate_content(prompt)
                corrected = completion.text
                st.text_area("✅ Corrected Output", corrected)

            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("🔁 Start Over"):
        st.session_state.clear()
        st.rerun()