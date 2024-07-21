
import os

# Set environment variables
os.environ['INPUT_CHOICE'] = 'Speech'
os.environ['CHATBOT_SERVICE'] = 'oogabooga'
os.environ['TTS_CHOICE'] = 'LOCAL_TTS' #
os.environ['VOICE_ID'] = 'pNInz6obpgDQGcFmaJgB' #
os.environ['CUDA_STATUS'] = 'True' #
os.environ['ELEVENLAB_KEY'] = 'sk_73c2d983c6ac7e9a1bbf6711bebd5e9d7a99cdde59904cf3' #
os.environ['VOICE_MODEL'] = 'Elli' #
os.environ['WHISPER_MODEL'] = 'base' #
os.environ['WHISPER_CHOICE'] = 'TRANSCRIBE' #
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['VTUBE_STUDIO_API_PORT'] = "8001" #
os.environ['GOOGLE_API_KEY'] = 'AIzaSyCEzc2NtaIa3eBMh5QNp1wDaeSCH0OrN-g'
# os.environ['OPENAI_API_KEY'] = "sk-************************************"

from utils.Model.GPT import *
from utils.Model.Gemini import *
from utils.processing import *

import threading
import utils.audio
import utils.vtube_studio
import logging 
import pyttsx3
import speech_recognition as sr
import os, sys, contextlib

from dotenv import load_dotenv
load_dotenv()

# Set up basic configuration for logging
logging.basicConfig(
    level=logging.ERROR,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Specify the format of log messages
    datefmt='%Y-%m-%d %H:%M:%S',  # Specify the format of the date in log messages
    handlers=[
        logging.FileHandler('app.log'),  # Log messages will be saved to a file named 'app.log'
        logging.StreamHandler()  # Log messages will also be output to the console
    ]
)

asr = sr.Recognizer()
TTS_CHOICE = os.environ.get("TTS_CHOICE")
TT_CHOICE = os.environ.get("WHISPER_CHOICE")
CHATBOT_CHOICE = os.environ.get("CHATBOT_SERVICE")
input_choice = os.environ.get("INPUT_CHOICE")
modelOptions = {
    'Gemini': 'gemini-1.5-flash',
    'Gemma2 9b': 'gemma2-9b-it',
    "Gemma 7b": 'gemma-7b-it',
    "Mixtral 8x7b": "mixtral-8x7b-32768",
    "LLaMA3 70b": "llama3-70b-8192",
    "LLaMA3 8b": "llama3-8b-8192",
}


@contextlib.contextmanager
def ignoreStderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)


def speech_to_text(audio: sr.AudioData):
    global asr
    try:
        return True, asr.recognize_google(audio, language="en")
    except Exception as e:
        return False, e.__class__


def main():

    selected_model_id = modelOptions.get('Gemini__', 0)

    if selected_model_id == 'gemini-1.5-flash':
        chatModel = Gemini()
    else:
        chatModel = Gemini()
    
    
    
    if input_choice.lower() == "speech":

        with ignoreStderr():

            with sr.Microphone() as source:
                asr.adjust_for_ambient_noise(source, duration=5)

                while True:
                    print('Say something')

                    audio=asr.listen(source)
                    is_valid_input, _input = speech_to_text(audio)

                    if is_valid_input:
                        message = chatModel.gemini_chat(_input)
                        message = postprocessing(message)
                        
                        engine = pyttsx3.init()
                        engine.setProperty('rate', 200)  
                        engine.say(message)
                        engine.runAndWait()

                        # # Set audio level using VTube Studio
                        # utils.vtube_studio.set_audio_level(0.5)

                        # # Play audio using VTube Studio
                        # utils.vtube_studio.speak()

                    else:
                        if _input is sr.RequestError:
                            print("No response from Google Speech Recognition service: {0}".format(_input))


def run_program():
    # Start the VTube Studio interaction in a separate thread
    vtube_studio_thread = threading.Thread(target=utils.vtube_studio.run_vtube_studio)
    print('vtube thread run successfully')
    vtube_studio_thread.daemon = True
    vtube_studio_thread.start()

    main()


if __name__ == "__main__":
    run_program()
