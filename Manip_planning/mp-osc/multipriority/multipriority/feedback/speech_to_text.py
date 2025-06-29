import whisper
import threading
from pydub import AudioSegment
from .detect_wakeword import DetectWord
import os

class Speech:
    """
    A class to handle analyzing spoken audio feedback using Whisper.

    Attributes:
        model (whisper.Model): The Whisper model used for analyzing the audio.
        feedback (str): The analyzed text extracted from the audio.
        audio_path (str): The directory path where the audio files are stored.
        update_event (threading.Event): An event that signals when a new recording is ready for analysis.
        feedback_queue (queue): The queue to store processes

    Methods:
        convert_wav_to_mp3(): Convert .wav to .mp3
        analyze(): Analyzes the recorded audio and returns the transcribed text.
        main(detect_word_instance): Orchestrates the saving and analyzing of audio using the DetectWord instance.
    """
    def __init__(self, model, audio_path, update_event, feedback_queue):
        """
        Initializes the Speech object with the specified filename and Whisper model.

        Parameters:
            model (str): The identifier for the Whisper model to be loaded.
            audio_path (str): The directory where audio recordings are stored.
            update_event (threading.Event): Event to signal when a new recording is available.
            feedback_queue ()
        """
        print('load_model' in dir(whisper))
        self.model = whisper.load_model(model)
        self.feedback = None
        self.audio_path = audio_path
        self.update_event = update_event
        self.filename = None
        self.feedback_queue = feedback_queue
        
    def convert_wav_to_mp3(self):
        """
        Converts a WAV file to an MP3 file.

        Parameters:
            filename (str): The file path of the WAV file to be converted.
        """
        # Load the WAV file
        audio_segment = AudioSegment.from_wav(self.filename)
        mp3_filename = self.filename.replace(".wav", ".mp3")
        audio_segment.export(mp3_filename, format="mp3")
        self.filename = mp3_filename

    def analyze(self):
        """
        Analyzes the recorded audio using the Whisper model and returns the transcribed text.

        Returns:
            str: The transcribed text from the audio analysis.
        """
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(self.filename)
        print("successfully loaded")
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device).float()

        # detect the spoken language
        _, probs = self.model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(self.model, mel, options)

        # print the recognized text
        print(result.text)
        return result.text

    def main(self, detect_word_instance):
        """
        Orchestrates the process of recording audio, saving it as an MP3 file, and analyzing it for transcribed text.
        
        Parameters:
            detect_word: An instance of DetectWord
        """
        while True:
            try:
                self.update_event.wait()  # Block until the event is set by DetectWord
                print("end waiting")
                self.filename = os.path.join(self.audio_path, detect_word_instance.filename + ".wav")
                self.convert_wav_to_mp3()
                input_text = self.analyze()
                self.feedback_queue.put(input_text)
                self.update_event.clear()  # Clear the event
            except KeyboardInterrupt:
                print("Speech analysis interrupted, exiting.")
                break



if __name__ == "__main__":
    # Create an event object for notifying about the new recording
    new_recording_event = threading.Event()

    # Define the directory for the audio files and the keywords
    import multipriority
    filepath = "/".join(multipriority.__path__[0].split('/')[:-1])
    # filepath = os.getcwd() + "/rcare_py/pyrcareworld/Examples"
    audio_path = filepath+"/feedback_audio"
    keywords = ["jarvis", "thank you"]
    access_key = os.environ.get("PORCUPINE_API_KEY")

    # Initialize DetectWord
    detect_word_instance = DetectWord(
        access_key= access_key,
        keywords=keywords,
        audio_directory=audio_path,
        update_event=new_recording_event,
        filepath=filepath
    )

    # Start the DetectWord instance in a separate thread
    detect_word_thread = threading.Thread(target=detect_word_instance.main)
    detect_word_thread.start()

    # Initialize Speech
    model_name = "base"
    import queue
    speech_instance = Speech(model_name, audio_path, new_recording_event, queue.Queue())

    # Start the Speech instance in a separate thread and pass the detect_word_instance to its main method
    speech_thread = threading.Thread(target=speech_instance.main, args=(detect_word_instance,))
    speech_thread.start()