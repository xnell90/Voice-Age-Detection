import argparse
import os

from pyannote.audio import Pipeline
from pydub import AudioSegment

parser = argparse.ArgumentParser(description='Speaker Diarization')

parser.add_argument("--input_directory", type=str, required=True)
parser.add_argument("--output_directory", type=str, required=True)

# speaker-diarization.yaml
parser.add_argument("--configuration_file", type=str, required=True)
parser.add_argument("--hugging_face_token", type=str, required=True)

args = parser.parse_args()

input_directory = args.input_directory
output_directory = args.output_directory

audio_filenames = os.listdir(input_directory)
audio_filepaths = [
  input_directory + audio_filename
  for audio_filename in audio_filenames
]

configuration_file = args.configuration_file
hugging_face_token = args.hugging_face_token

speaker_diarization_pipeline = Pipeline.from_pretrained(
  configuration_file,
  use_auth_token=hugging_face_token
)

for audio_filename, audio_filepath in zip(audio_filenames, audio_filepaths):
  speaker_diarization = speaker_diarization_pipeline(audio_filepath)

  speaker_segments = []
  for segment, _, speaker in speaker_diarization.itertracks(yield_label=True):
    speaker_segment = {
      "id": speaker,
      "start": segment.start,
      "end": segment.end
    }
    speaker_segments.append(speaker_segment)

  sound = AudioSegment.from_file(audio_filepath)

  for index, speaker_segment in enumerate(speaker_segments):
    id = speaker_segment['id']

    start = speaker_segment['start']
    start = start * 1000

    end = speaker_segment['end']
    end = end * 1000

    segment_filename = audio_filename[:-4] + "_%d_%s_.mp3" % (index, id)
    segment_filepath = output_directory + segment_filename
    sound[start:end].export(segment_filepath, format='mp3')
