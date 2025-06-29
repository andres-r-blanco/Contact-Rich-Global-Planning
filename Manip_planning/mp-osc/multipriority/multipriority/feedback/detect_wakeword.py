import pvporcupine as pv
from pvrecorder import PvRecorder
import threading
import os
import wave
import time
import struct

class DetectWord:
    """
    A class to detect specific wake words using Porcupine and record audio upon detection.

    Attributes:
        access_key (str): The access key for using Porcupine's API.
        keywords (list): List of keywords (wake words) to be detected.
        audio_directory (str): Directory to save the recorded audio files.
        wav_file (wave.Wave_write): File object for the recorded audio.

    Methods:
        start_record(name): Starts recording audio and saves it to a file.
        stop_record(): Stops the audio recording and closes the file.
        main(): Main loop for continuously listening and processing audio for wake word detection.
    """
    def __init__(self, access_key, keywords, audio_directory, update_event, filepath):
        """
        Initializes the DetectWord object with the provided access key and keywords.

        Parameters:
            access_key (str): The access key for using Porcupine's API.
            keywords (list): List of keywords (wake words) to be detected.
            audio_directory (str): The directory where audio recordings will be saved.
            update_event (threading.Event): Event to signal when a new recording is available.
            filepath (str): Path to the keyword file for Porcupine.
        """
        self.access_key = access_key
        self.keywords = keywords
        keyword_paths = [pv.KEYWORD_PATHS[keywords[0]], filepath + "/third_party/Thank-you_en_linux_v3_0_0.ppn"]
        sensitivities = [0.5] * len(keyword_paths)
        self.porcupine = pv.create(
            access_key=self.access_key,
            keyword_paths=keyword_paths,
            sensitivities=sensitivities)
        self.wav_file = None
        self.filename = None
        self.audio_directory = audio_directory
        self.update_event = update_event
        self.command = 0

    def start_record(self, name):
        """
        Starts recording audio and saves it to a file with the given name.

        Parameters:
            name (str): The name of the file to save the audio recording.
        """
        filepath = os.path.join(self.audio_directory, name)
        self.wav_file = wave.open(filepath, "w")
        self.wav_file.setnchannels(1)
        self.wav_file.setsampwidth(2)
        self.wav_file.setframerate(16000)

    def stop_record(self):
        """
        Stops the audio recording and closes the file.
        """
        self.wav_file.close()
        self.wav_file = None

    def main(self):
        """
        Main loop for continuously listening to audio input and processing it for wake word detection.
        Recording starts when the first keyword is detected and stops when the second keyword is detected.
        """
        audio_device_index = -1
        recorder = PvRecorder(frame_length=self.porcupine.frame_length, device_index=audio_device_index)
        recorder.start()

        print('Listening for keywords... (press Ctrl+C to exit)')

        is_recording = False

        try:
            while True:
                pcm = recorder.read()
                result = self.porcupine.process(pcm)

                if result == 0:  # Start recording on detecting the first keyword
                    if not is_recording:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        self.filename = f"feedback_{timestamp}"
                        print(f'Detected {self.keywords[0]}, start recording to {self.filename+".wav"}')
                        is_recording = True
                        self.start_record(self.filename+".wav")

                elif result == 1:  # Stop recording on detecting the second keyword
                    if is_recording:
                        print(f'Detected {self.keywords[1]}, stop recording')
                        is_recording = False
                        self.stop_record()
                        self.update_event.set()

                if is_recording and self.wav_file is not None:
                    self.wav_file.writeframes(struct.pack("h" * len(pcm), *pcm))

        except KeyboardInterrupt:
            print('Stopping ...')
        finally:
            if self.wav_file is not None:
                self.wav_file.close()
            recorder.delete()
            self.porcupine.delete()

if __name__ == '__main__':
    # access_key = "${ACCESS_KEY}"
    import multipriority
    filepath = "/".join(multipriority.__path__[0].split('/')[:-1])
    
    audio_path = "/tmp/feedback_audio"
    if not os.path.exists(audio_path):
        os.makedirs(audio_path)
    update_event = threading.Event()
    access_key = os.environ.get("PORCUPINE_API_KEY")
    dw = DetectWord(access_key, ["jarvis", "thank you"], audio_path, update_event, filepath)
    detect_word_thread = threading.Thread(target=dw.main)
    detect_word_thread.start()
