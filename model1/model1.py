import os
import warnings

warnings.filterwarnings(action='ignore')

import pandas as pd
import librosa
import numpy as np

from sklearn.utils import shuffle
from PIL import Image
from tqdm import tqdm

# Global vars
RANDOM_SEED = 1337
SAMPLE_RATE = 32000
SIGNAL_LENGTH = 5  # seconds
SPEC_SHAPE = (224, 224)  # height x width
FMIN = 20
FMAX = 16000

# Code adapted from:
# https://www.kaggle.com/frlemarchand/bird-song-classification-using-an-efficientnet
# Make sure to check out the entire notebook.

# Load metadata file
train = pd.read_csv('../input/birdclef-2021/train_metadata.csv', )
# Second, assume that birds with the most training samples are also the most common
# A species needs at least 200 recordings with a rating above 4 to be considered common
birds_count = {}
for bird_species, count in zip(train.primary_label.unique(),
                               train.groupby('primary_label')['primary_label'].count().values):
    birds_count[bird_species] = count
most_represented_birds = [key for key, value in birds_count.items()]

TRAIN = train.query('primary_label in @most_represented_birds')
LABELS = sorted(TRAIN.primary_label.unique())

# Let's see how many species and samples we have left
print('NUMBER OF SPECIES IN TRAIN DATA:', len(LABELS))
print('NUMBER OF SAMPLES IN TRAIN DATA:', len(TRAIN))
print('LABELS:', most_represented_birds)
# Shuffle the training data and limit the number of audio files to MAX_AUDIO_FILES
TRAIN = shuffle(TRAIN, random_state=RANDOM_SEED)


# Define a function that splits an audio file,
# extracts spectrograms and saves them in a working directory
def get_spectrograms(filepath, primary_label, output_dir):
    # Open the file with librosa (limited to the first 15 seconds)
    sig, rate = librosa.load(filepath, sr=SAMPLE_RATE, offset=None, duration=15)

    # Split signal into five second chunks
    sig_splits = []
    for i in range(0, len(sig), int(SIGNAL_LENGTH * SAMPLE_RATE)):
        split = sig[i:i + int(SIGNAL_LENGTH * SAMPLE_RATE)]

        # End of signal?
        if len(split) < int(SIGNAL_LENGTH * SAMPLE_RATE):
            break

        sig_splits.append(split)

    # Extract mel spectrograms for each audio chunk
    s_cnt = 0
    saved_samples = []
    for chunk in sig_splits:

        hop_length = int(SIGNAL_LENGTH * SAMPLE_RATE / (SPEC_SHAPE[1] - 1))
        mel_spec = librosa.feature.melspectrogram(y=chunk,
                                                  sr=SAMPLE_RATE,
                                                  n_fft=2048,
                                                  hop_length=hop_length,
                                                  n_mels=SPEC_SHAPE[0],
                                                  fmin=FMIN,
                                                  fmax=FMAX)

        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        mel_spec -= mel_spec.min()
        mel_spec /= mel_spec.max()

        # Save as image file
        save_dir = os.path.join(output_dir, primary_label)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, filepath.rsplit(os.sep, 1)[-1].rsplit('.', 1)[0] +
                                 '_' + str(s_cnt) + '.png')
        im = Image.fromarray(mel_spec * 255.0).convert("L")
        im.save(save_path)

        saved_samples.append(save_path)
        s_cnt += 1

    return saved_samples


print('FINAL NUMBER OF AUDIO FILES IN TRAINING DATA:', len(TRAIN))
# Parse audio files and extract training samples
input_dir = '../input/birdclef-2021/train_short_audio/'
output_dir = '../working/melspectrogram_dataset/'
samples = []
with tqdm(total=len(TRAIN)) as pbar:
    for idx, row in TRAIN.iterrows():
        pbar.update(1)

        if row.primary_label in most_represented_birds:
            audio_file_path = os.path.join(input_dir, row.primary_label, row.filename)
            samples += get_spectrograms(audio_file_path, row.primary_label, output_dir)
print(samples)
str_samples = ','.join(samples)
TRAIN_SPECS = shuffle(samples, random_state=RANDOM_SEED)
filename = open('a.txt', 'w')
filename.write(str_samples)
filename.close()