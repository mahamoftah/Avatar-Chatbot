
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

import colorama
import humanize, os, threading
import utils.audio
import utils.vtube_studio
import logging 

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

def main():
    chat_history = []
    selected_model_id = modelOptions.get('Gemini__', 0)

    if selected_model_id == 'gemini-1.5-flash':
        chatModel = Gemini()
    else:
        chatModel = Gemini()
    
    
    
    if input_choice.lower() == "speech":
        import utils.transcriber_translate
        import utils.hotkeys
        while True:
            print("You" + colorama.Fore.GREEN + colorama.Style.BRIGHT + " (mic) " + colorama.Fore.RESET + ">", end="",
                  flush=True)
            utils.hotkeys.audio_input_await()
            print(
                "\rYou" + colorama.Fore.GREEN + colorama.Style.BRIGHT + " (mic " + colorama.Fore.YELLOW + "[Recording]" + colorama.Fore.GREEN + ") " + colorama.Fore.RESET + ">",
                end="", flush=True)
            audio_buffer = utils.audio.record()

            try:
                tanscribing_log = "\rYou" + colorama.Fore.GREEN + colorama.Style.BRIGHT + " (mic " + colorama.Fore.BLUE + "[Transcribing (" + str(
                    humanize.naturalsize(
                        os.path.getsize(audio_buffer))) + ")]" + colorama.Fore.GREEN + ") " + colorama.Fore.RESET + "> "
                logging.info(tanscribing_log)
                if TT_CHOICE.upper() == "TRANSLATE":
                    transcript = utils.transcriber_translate.translate_any_to_english(audio_buffer)

                elif TT_CHOICE.upper() == "TRANSCRIBE":
                    transcript = utils.transcriber_translate.to_transcribe_original_language(audio_buffer)
                chat_history.append({'role': 'user', 'content' : transcript})
            except Exception as e:
                logging.error(e)
                continue

            # Clear the last line.
            logging.info(tanscribing_log)
            print("\rYou" + colorama.Fore.GREEN + colorama.Style.BRIGHT + " (mic) " + colorama.Fore.RESET + "> ",
                  end="", flush=True)

            logging.info(transcript.strip())
            combined_input = transcript
            
            if CHATBOT_CHOICE == "oogabooga":
                message = chatModel.gemini_chat(combined_input)
                message = postprocessing(message)
            else:
                logging.error("Sorry Wrong Chatbot Choice")
            
                
            chat_history.append({'role': 'bot', 'content' : message})
            if TTS_CHOICE == "GTTS_API":
                import utils.GTTS_API
                utils.GTTS_API.generate_voice(message)
            
            if TTS_CHOICE == "GTTS_API":
                import utils.Elevenlabs
                utils.Elevenlabs.generate_voice(message)

            elif TTS_CHOICE == "LOCAL_TTS":
                import utils.Offline_tts
                utils.Offline_tts.voice_generation(message)
            else:
                logging.error("The Choice put in .env file not correct!")

            # Set audio level using VTube Studio
            utils.vtube_studio.set_audio_level(0.5)

            # Play audio using VTube Studio
            utils.vtube_studio.speak()

            # After use, delete the recording.
            try:
                os.remove(audio_buffer)
            except:
                pass

    if input_choice.lower() == "text":
        while True:

            print( colorama.Fore.GREEN + colorama.Style.BRIGHT + "YOU : ", end="", flush=True)
            transcript = input(colorama.Fore.GREEN + colorama.Style.BRIGHT + colorama.Fore.RESET + ">")

            if CHATBOT_CHOICE == "oogabooga":
                message = ''
                for response in chatModel.generate(combined_input, 'en'):
                    if response is None:
                        break
                    message += response
            else:
                logging.error("Sorry Wrong Chatbot Choice")
           
            message = postprocessing(message)

            if TTS_CHOICE == "ELEVENLABS":
                import utils.GTTS_API
                utils.GTTS_API.generate_voice(message)

            #LOCAL_TTS is out of support for now. Will be back soon.
            elif TTS_CHOICE == "LOCAL_TTS":
                import utils.Offline_tts
                utils.Offline_tts.voice_generation(message)

            if TTS_CHOICE == "GTTS_API":
                import utils.Elevenlabs
                utils.Elevenlabs.generate_voice(message)
            else:
                logging.error("The Choice put in .env file not correct!")

            # Set audio level using VTube Studio
            utils.vtube_studio.set_audio_level(0.5)

            # Play audio using VTube Studio
            utils.vtube_studio.speak()

            # After use, delete the recording.
            try:
                os.remove(audio_buffer)
            except:
                pass


def run_program():
    # Start the VTube Studio interaction in a separate thread
    vtube_studio_thread = threading.Thread(target=utils.vtube_studio.run_vtube_studio)
    print('vtube thread run successfully')
    vtube_studio_thread.daemon = True
    vtube_studio_thread.start()

    main()


if __name__ == "__main__":

    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Create the resource directory path based on the current directory
    resource_directory = os.path.join(current_directory, "utils","resource")
    os.makedirs(resource_directory, exist_ok=True)

    # Create the voice_in and voice_out directory paths
    voice_in_directory = os.path.join(resource_directory, "voice_in")
    voice_out_directory = os.path.join(resource_directory, "voice_out")

    # Create the voice_in and voice_out directories if they don't exist
    os.makedirs(voice_in_directory, exist_ok=True)
    os.makedirs(voice_out_directory, exist_ok=True)


    run_program()
