#This script converts mp3 files to wav files
import argparse
import os

from pydub import AudioSegment

parser = argparse.ArgumentParser(description='Convert MP3 files to WAV files')

parser.add_argument("--input_directory", type=str, required=True)
parser.add_argument("--output_directory", type=str, required=True)

args = parser.parse_args()

input_directory = args.input_directory
output_directory = args.output_directory

mp3_filenames = os.listdir(input_directory)
mp3_filepaths = [
  input_directory + mp3_filename
  for mp3_filename in mp3_filenames
]

wav_filenames = [
  mp3_filename[:-4] + '.wav'
  for mp3_filename in mp3_filenames
]
wav_filepaths = [
  output_directory + wav_filename
  for wav_filename in wav_filenames
]

for mp3_filepath, wav_filepath in zip(mp3_filepaths, wav_filepaths):
  sound = AudioSegment.from_file(mp3_filepath)
  sound.export(wav_filepath, format='wav')
