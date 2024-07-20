import os
from gtts import gTTS


from dotenv import load_dotenv
load_dotenv()



current_directory = os.path.dirname(os.path.abspath(__file__))
FILENAME = "output.mp3"
OUTPUT_PATH = os.path.join(current_directory, "resource", "voice_out", FILENAME)




def generate_voice(responded_text):
    tts = gTTS(responded_text, lang='en')
    tts.save(OUTPUT_PATH)
    return OUTPUT_PATH




