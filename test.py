import sys
import contextlib
import numpy as np
import pyaudio
import wave
import speech_recognition as sr
from io import BytesIO
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from gtts import gTTS
from pydub import AudioSegment
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import re
import os

# ===========================================================================================================================================

class Gemini:

    def __init__(self):
        # self.messages = []
        self.messages = [{"role": "user", "parts": ["Yor are helpfull assistant, Please don't mention that you take your response from \
                provided information in any way because i will use these responses to answer questions related to my business as \
                if i was the one answering"]}]
        os.environ['GOOGLE_API_KEY']="AIzaSyBaz13UTSLEsag18c_rHQ9yFUbX4sx3YYM"
        GOOGLE_API_KEY="AIzaSyBaz13UTSLEsag18c_rHQ9yFUbX4sx3YYM"
        genai.configure(api_key=GOOGLE_API_KEY)
        os.environ['PINECONE_API_KEY'] = 'c8519fb3-b5b7-461e-afa7-0090e4a5fa43'
        self.vector_db = self.load_vector_db()
        self.chat = genai.GenerativeModel('gemini-1.5-flash').start_chat(history=[])

    def load_vector_db(self):
        return FAISS.load_local("faiss_index", self.embedding_google(), allow_dangerous_deserialization=True)

    def embedding_google(self):
        return GoogleGenerativeAIEmbeddings(model='models/embedding-001', task_type="SEMANTIC_SIMILARITY")

    def search_similar_context(self, vector_db, question, n):
        if vector_db:
            docs = vector_db.similarity_search(question, k=n)
            return docs


    def process_question(self, question):
        response = genai.GenerativeModel('gemini-1.5-flash').generate_content(question, stream=False, generation_config=genai.GenerationConfig(
        max_output_tokens=60))
        if response is not None:
            return response.text
        else:
            return "Ask you question again"


    def generate(self, question):

        similar_text = ""
        similar_context = self.search_similar_context(self.vector_db, question, 5)
        for context in similar_context:
            similar_text += context.page_content

        # i wanted to pass the context as system prompt but unfortunately there is only two roles [user and model].
        # so i thought that it's posible to pass it as a seperated user prompt, or append it to the input question to pass it as cominaed user prompt
        # prompt = f"Use the given below context to answer the user query. {similar_text} Question: {question}?"
        # or self.messages.append({"role": "user", "parts": [similar_text]})
       
        self.messages.append({"role": "user", "parts": [similar_text]})
        self.messages.append({"role": "user", "parts": [question]})
        
        response = self.process_question(self.messages)
        self.messages.append({"role": "model", "parts": [response]})

        return response
    
    def gemini_chat(self, question):

        similar_text = ""
        similar_context = self.search_similar_context(self.vector_db, question, 5)
        for context in similar_context:
            similar_text += context.page_content

        self.chat.send_message(similar_text)
        response = self.chat.send_message(question)
        
        return response.text
    
# ===========================================================================================================================================

def postprocessing(text):

        pattern = re.compile(r'https?://\S+|www\.\S+')
        text = pattern.sub(' ', text)

        pattern = re.compile(r'[*#,!?$@:;]')
        text = pattern.sub('', text)

        for pattern in ['Based on the information you provided,', 'The text you provided gives', 'The provided text doesn\'t specify', 'The provided text mentions that', 'Based on the provided text', 'Based on the provided text,', 'According to the provided text', 'The provided text states that', 'The provided text', 'The text you provided lists', 'The text you provided', 'The information you provided']:
            text = re.sub(pattern, '' , text)
        
        text = re.sub(r'\.[^.]*$', '', text)

        pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F"
            "\U0001FA70-\U0001FAFF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        
        text = pattern.sub(' ', text)

        return text.strip()

# ===========================================================================================================================================

asr = sr.Recognizer()
model = Gemini()

# ===========================================================================================================================================

class SpeechRecognitionThread(QThread):
    recognized_text = pyqtSignal(str)
    audio_data = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.running = True

    def run(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=5)
            while self.running:
                try:
                    audio = self.recognizer.listen(source)
                    text = self.recognizer.recognize_google(audio, language="en")
                    self.recognized_text.emit(text)
                    self.audio_data.emit(np.frombuffer(audio.get_raw_data(), dtype=np.int16))
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    self.recognized_text.emit(f"Request error: {e}")

    def stop(self):
        self.running = False
        self.wait()

# ===========================================================================================================================================

class WaveformWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.audio_data = np.zeros(1024)
        self.setMinimumHeight(200)
        
    def update_waveform(self, audio_data):
        self.audio_data = audio_data
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(QColor(0, 0, 255))
        pen.setWidth(2)
        painter.setPen(pen)
        
        width = self.width()
        height = self.height()
        middle = height // 2
        scale = height / 32768 / 2
        
        painter.drawLine(0, middle, width, middle)  # Draw a center line

        for i in range(1, len(self.audio_data)):
            x1 = int((i - 1) * width / len(self.audio_data))
            y1 = int(middle - (self.audio_data[i - 1] * scale))
            x2 = int(i * width / len(self.audio_data))
            y2 = int(middle - (self.audio_data[i] * scale))
            painter.drawLine(x1, y1, x2, y2)

# ===========================================================================================================================================

class AudioPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.speech_thread = SpeechRecognitionThread()
        self.speech_thread.recognized_text.connect(self.process_text)
        self.speech_thread.audio_data.connect(self.waveform_widget.update_waveform)
        self.speech_thread.start()

    def initUI(self):
        self.setWindowTitle('Speech to Speech Avatar')
        self.setGeometry(300, 300, 800, 400)

        # Main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Waveform display widget
        self.waveform_widget = WaveformWidget()
        self.layout.addWidget(self.waveform_widget)

        self.show()

    def process_text(self, text):
        print(f'{text}\n')
        response = self.generate_response(text)
        print(f'{response}\n')
        self.play_audio(response)

    def generate_response(self, text):
        response = model.generate(text)
        response = postprocessing(response)
        return response

    def play_audio(self, message):
        sound_file = BytesIO()
        tts = gTTS(message, lang="en")
        tts.write_to_fp(sound_file)
        sound_file.seek(0)
        
        # Convert MP3 to WAV
        mp3_audio = AudioSegment.from_file(sound_file, format="mp3")
        wav_io = BytesIO()
        mp3_audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # Open the WAV file with wave
        wf = wave.open(wav_io, 'rb')
        self.p = pyaudio.PyAudio()  # Store PyAudio instance in the class
        self.stream = self.p.open(format=self.p.get_format_from_width(wf.getsampwidth()),
                                  channels=wf.getnchannels(),
                                  rate=wf.getframerate(),
                                  output=True)

        # Setup timer for updating the UI
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.update_ui(wf))
        self.timer.start(30)

    def update_ui(self, wf):
        chunk = 1024
        data = wf.readframes(chunk)
        if data:
            self.stream.write(data)
            audio_data = np.frombuffer(data, dtype=np.int16)
            self.waveform_widget.update_waveform(audio_data)
        else:
            self.timer.stop()
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

    def closeEvent(self, event):
        self.speech_thread.stop()
        event.accept()

# ===========================================================================================================================================

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioPlayer()
    sys.exit(app.exec_())