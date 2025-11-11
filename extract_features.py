'''
import librosa
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from librosa.feature import spectral_centroid, spectral_bandwidth, spectral_rolloff
from tqdm import tqdm

new_columns = [
  'mean_spectral_bandwidth',
  'mean_spectral_centroid',
  'mean_spectral_rolloff'
]
rearranged_columns = [
  'filename',
  'up_votes',
  'down_votes',
  'locale',
  'gender',
  'mean_spectral_bandwidth',
training_df = pd.read_csv("./datasets/common_voice/training.csv")
training_filenames = list(training_df.filename)
training_filepaths = [
  './datasets/common_voice/clips/%s' % training_filename
  for training_filename in training_filenames
]

validation_df = pd.read_csv("./datasets/common_voice/validation.csv")
validation_filenames = list(validation_df.filename)
validation_filepaths = [
  './datasets/common_voice/clips/%s' % validation_filename
  for validation_filename in validation_filenames
]

filepaths = training_filepaths + validation_filepaths
  'mean_spectral_centroid',
  'mean_spectral_rolloff',
  'age'
]

def feature_extraction(filename):
  filepath = './datasets/common_voice/clips/%s' % filename
  y, sr = librosa.load(filepath)

  return pd.Series(
    [
      np.mean(spectral_bandwidth(y, sr)),
      np.mean(spectral_centroid(y, sr)),
      np.mean(spectral_rolloff(y, sr))
    ]
  )

training_df = pd.read_csv("./datasets/common_voice/training.csv")
training_df[new_columns] = training_df['filename'].apply(feature_extraction)
training_df = training_df[rearranged_columns]
training_df.to_csv('./datasets/common_voice/training(1).csv', index=False)

validation_df = pd.read_csv("./datasets/common_voice/validation.csv")
validation_df[new_columns] = validation_df['filename'].apply(feature_extraction)
validation_df = validation_df[rearranged_columns]
validation_df.to_csv('./datasets/common_voice/validation(1).csv', index=False)

testing_df = pd.read_csv("./datasets/common_voice/testing.csv")
testing_df[new_columns] = testing_df['filename'].apply(feature_extraction)
testing_df = testing_df[rearranged_columns]
testing_df.to_csv('./datasets/common_voice/testing(1).csv', index=False)
'''
