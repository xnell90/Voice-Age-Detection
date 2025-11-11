import matplotlib.pyplot as plt
import numpy as np

from librosa import load, power_to_db
from librosa.display import specshow
from librosa.feature import melspectrogram
from pydub import AudioSegment

KEYS = ['under-20', '20-29', '30-39', '40-49', '50-59', '60+']

def age_distribution(dataframe):
  age_category_counter = dataframe.age.value_counts()
  age_category_counter = dict(age_category_counter)
  age_category_counter = {
    key: age_category_counter[key]
    for key in KEYS
  }

  values = list(age_category_counter.values())

  plt.figure(figsize=(15,10))
  plt.title("Age Category Distribution")
  plt.bar(range(6), values, tick_label=KEYS)
  plt.show()

def audio_duration(mp3_filepath):
  audio = AudioSegment.from_mp3(mp3_filepath)
  return audio.duration_seconds

def common_voice_age_distribution_by_gender(dataframe):
  male_filter = dataframe.gender == 'male'
  male_age_category_counter = dataframe[male_filter].age.value_counts()
  male_age_category_counter = dict(male_age_category_counter)
  male_age_category_counter = {
    key: male_age_category_counter[key]
    for key in KEYS
  }

  male_values = np.array(list(male_age_category_counter.values()))

  female_filter = dataframe.gender == 'female'
  female_age_category_counter = dataframe[female_filter].age.value_counts()
  female_age_category_counter = dict(female_age_category_counter)
  female_age_category_counter = {
    key: female_age_category_counter[key]
    for key in KEYS
  }

  female_values = np.array(list(female_age_category_counter.values()))

  unknown_gender_filter = dataframe.gender == '?'
  unknown_gender_age_category_counter = dataframe[unknown_gender_filter].age.value_counts()
  unknown_gender_age_category_counter = dict(unknown_gender_age_category_counter)
  unknown_gender_age_category_counter = {
    key: unknown_gender_age_category_counter[key]
    for key in KEYS
  }

  unknown_gender_values = np.array(list(unknown_gender_age_category_counter.values()))

  other_filter = dataframe.gender == 'other'
  other_age_category_counter = dataframe[other_filter].age.value_counts()
  other_age_category_counter = dict(other_age_category_counter)
  other_age_category_counter = {
    key: other_age_category_counter[key]
    for key in KEYS
  }

  other_values = np.array(list(other_age_category_counter.values()))

  plt.figure(figsize=(15,10))
  plt.title("Age Category Distribution By Gender")
  plt.bar(range(6), male_values, tick_label=KEYS, color='blue')
  plt.bar(
    range(6),
    female_values,
    bottom=male_values,
    tick_label=KEYS,
    color='pink'
  )
  plt.bar(
    range(6),
    other_values,
    bottom=(female_values + male_values),
    tick_label=KEYS,
    color='gold'
  )
  plt.bar(
    range(6),
    unknown_gender_values,
    bottom=(other_values + female_values + male_values),
    tick_label=KEYS,
    color='yellow'
  )
  plt.legend(['male', 'female', 'other', '?'])
  plt.show()

def display_melspectrogram(mp3_filepath):
  y, sr = load(mp3_filepath)

  melspectrogram_db = melspectrogram(y=y, sr=sr)
  melspectrogram_db = power_to_db(melspectrogram_db, ref=np.max)

  fig, ax = plt.subplots(figsize=(15, 10))
  img = specshow(melspectrogram_db, x_axis='time', y_axis='mel', sr=sr, ax=ax)
  fig.colorbar(img, ax=ax, format='%+2.0f dB')

  ax.set(title='Mel-frequency Spectrogram For %s' % mp3_filepath)
  plt.show()

def display_modified_melspectrogram(np_filepath):
  modified_melpsectrogram = np.load(np_filepath)

  plt.figure(figsize=(15, 10))
  plt.imshow(modified_melpsectrogram, cmap='hot')
  plt.title("Modified Mel-frequecy Spectrogram for %s" % np_filepath)
  plt.colorbar()

  plt.show()

def modify_melspectrogram(mp3_filepath):
  y, sr = load(mp3_filepath)

  melspectrogram_db = melspectrogram(y, sr)
  melspectrogram_db = power_to_db(melspectrogram_db, ref=np.max)
  melspectrogram_db = np.resize(melspectrogram_db.T, (431, 128)).T
  melspectrogram_db = melspectrogram_db.reshape(
    *melspectrogram_db.shape, 1
  )

  return (melspectrogram_db + 80) / 80

def speech_accent_age_distribution_by_accent(dataframe):
  english_filter = dataframe.accent == 'english'
  english_age_category_counter = dataframe[english_filter].age.value_counts()
  english_age_category_counter = dict(english_age_category_counter)
  english_age_category_counter = {
    key: english_age_category_counter[key]
    for key in KEYS
  }

  english_values = np.array(list(english_age_category_counter.values()))

  spanish_filter = dataframe.accent == 'spanish'
  spanish_age_category_counter = dataframe[spanish_filter].age.value_counts()
  spanish_age_category_counter = dict(spanish_age_category_counter)
  spanish_age_category_counter = {
    key: spanish_age_category_counter[key]
    for key in KEYS
  }

  spanish_values = np.array(list(spanish_age_category_counter.values()))

  arabic_filter = dataframe.accent == 'arabic'
  arabic_age_category_counter = dataframe[arabic_filter].age.value_counts()
  arabic_age_category_counter = dict(arabic_age_category_counter)
  arabic_age_category_counter = {
    key: arabic_age_category_counter[key]
    for key in KEYS
  }

  arabic_values = np.array(list(arabic_age_category_counter.values()))

  plt.figure(figsize=(15,10))
  plt.title("Age Category Distribution By Accent")
  plt.bar(range(6), english_values, tick_label=KEYS, color='green')
  plt.bar(range(6), spanish_values, bottom=english_values, tick_label=KEYS, color='orange')
  plt.bar(range(6), arabic_values, bottom=english_values+spanish_values, tick_label=KEYS, color='red')
  plt.legend(['english', 'spanish', 'arabic'])
  plt.show()

def speech_accent_age_distribution_by_gender(dataframe):
  male_filter = dataframe.gender == 'male'
  male_age_category_counter = dataframe[male_filter].age.value_counts()
  male_age_category_counter = dict(male_age_category_counter)
  male_age_category_counter = {
    key: male_age_category_counter[key]
    for key in KEYS
  }

  male_values = np.array(list(male_age_category_counter.values()))

  female_filter = dataframe.gender == 'female'
  female_age_category_counter = dataframe[female_filter].age.value_counts()
  female_age_category_counter = dict(female_age_category_counter)
  female_age_category_counter = {
    key: female_age_category_counter[key]
    for key in KEYS
  }

  female_values = np.array(list(female_age_category_counter.values()))

  plt.figure(figsize=(15,10))
  plt.title("Age Category Distribution By Gender")
  plt.bar(range(6), male_values, tick_label=KEYS, color='blue')
  plt.bar(range(6), female_values, bottom=male_values, tick_label=KEYS, color='pink')
  plt.legend(['male', 'female'])
  plt.show()
