from TTS.api import TTS
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the file in the 'resource' folder
FILE_PATH = os.path.join(script_dir, "resource", "voice_out", "local_tts_output.wav")

# List available 🐸TTS models and choose the first one
# Useable models: ['tts_models/en/ek1/tacotron2', 'tts_models/en/ljspeech/tacotron2-DDC', 
# 'tts_models/en/ljspeech/tacotron2-DDC_ph', 'tts_models/en/ljspeech/glow-tts', 
# 'tts_models/en/jenny/jenny']

def voice_generation(responded_text, model_name='tts_models/en/ljspeech/tacotron2-DDC_ph'):
    # Instantiate the TTS model
    tts = TTS(model_name=model_name, progress_bar=True, gpu=False)
    
    # Generate the TTS output and save it to a file
    tts.tts_to_file(text=responded_text, emotion="happy", file_path=FILE_PATH)
    return FILE_PATH

# # Example usage:
# responded_text = "Hello, this is a test."
# generated_file_path = voice_generation(responded_text)
# print(f"Audio file saved at: {generated_file_path}")
