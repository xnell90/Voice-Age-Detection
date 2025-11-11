import joblib
import numpy as np
import pandas as pd
import warnings

from librosa import load, power_to_db
from librosa.feature import melspectrogram

def compute_melspectrogram(mp3_filepath):
  warnings.filterwarnings("ignore", category=UserWarning)
  y, sr = load(mp3_filepath)

  np_filename = mp3_filepath.split("/")[-1][:-4] + ".npy"
  np_filepath = './datasets/speech_accent_archive/melspectrograms/%s' % np_filename

  melspectrogram_db = melspectrogram(y, sr)
  melspectrogram_db = power_to_db(melspectrogram_db, ref=np.max)
  melspectrogram_db = np.resize(melspectrogram_db.T, (431, 128)).T
  melspectrogram_db = melspectrogram_db.reshape(
    *melspectrogram_db.shape, 1
  )
  melspectrogram_db = (melspectrogram_db + 80) / 80

  np.save(np_filepath, melspectrogram_db)

dataframe = pd.read_csv("./datasets/speech_accent_archive/metadata(1).csv")
filenames = list(dataframe.filename)
filepaths = [
  './datasets/speech_accent_archive/clips/%s' % filename
  for filename in filenames
]

jobs = [
  joblib.delayed(compute_melspectrogram)(filepath)
  for filepath in filepaths
]
joblib.Parallel(n_jobs=12, verbose=1)(jobs)
