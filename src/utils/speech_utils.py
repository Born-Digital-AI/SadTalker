from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def get_audio_len_sec(audio_path):
    audio = AudioSegment.from_file(audio_path)
    duration_ms = len(audio)
    return round(duration_ms / 1000.0, 2)


def get_speech_start_sec(audio_path):
    audio = AudioSegment.from_file(audio_path)
    nonsilent_data = detect_nonsilent(audio, min_silence_len=200, silence_thresh=-16)
    first_nonsilent_part = nonsilent_data[0] if nonsilent_data else None

    if first_nonsilent_part:
        speech_start_sec = round(first_nonsilent_part[0] / 1000.0, 2)
    else:
        speech_start_sec = 0.00

    return speech_start_sec
