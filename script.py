import argparse
import tensorflow as tf
from IPython import display
import numpy as np

def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


class Recognizer:
    def __init__(self):
        self.model = None

    def model_load(self, model_path):
        # Load the model
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, audio_file):
        if self.model is None:
            raise ValueError("Model has not been loaded. Call model_load() first.")

        audio_file, sample_rate = tf.audio.decode_wav(audio_file, desired_channels=1, desired_samples=4*16000,)
        audio_file = tf.squeeze(audio_file, axis=-1)

        # waveform = audio_file
        # display.display(display.Audio(waveform, rate=16000))

        audio_file = get_spectrogram(audio_file)
        audio_file = audio_file[tf.newaxis,...]

        prediction = self.model.predict(audio_file)
        prediction = np.argmax(tf.nn.softmax(prediction[0]))
        prediction = audio_labels[prediction]
        print(prediction)
        return prediction


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_files", nargs='+', help="Paths to the audio files")
    parser.add_argument("model_path", help="Path to the saved model")
    args = parser.parse_args()

    model_path = args.model_path
    audio_files = args.audio_files

    # Instantiate Recognizer class
    recognizer = Recognizer()

    # Load the model
    recognizer.model_load(model_path)

    # Audio labels
    audio_labels = ['oragh','felezat', 'sandogh sahami', 'sandogh daramad sabet', 'sandogh mokhtalet', 'sandogh ghabele moamele', 'arz', 'sekke', 'bank', 'tala' , 'naft', 'moshtagaht']
    
    # List to store predicted labels
    predicted_labels = []


    # Predict the labels for each audio file
    for audio_file in audio_files:
        audio_file = tf.io.read_file(str(audio_file))

        predicted_label = recognizer.predict(audio_file)
        predicted_labels.append(predicted_label)
    
    print(predicted_labels)