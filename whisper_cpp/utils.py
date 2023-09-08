import os
import wave
import sys
import math
import numpy as np

from . import whisper_cpp

class suppress_stdout_stderr(object):
    # Oddly enough this works better than the contextlib version
    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()

def read_wav(fname: str, pcmf32: list, pcmf32s: list, stereo: bool) -> bool:
    """read wav file

    Args:
        fname (str): input wav file name (with path)
        pcmf32 (list): pcm float32 data
        pcmf32s (list): stereo pcm float32 data
        stereo (bool): whether to read stereo data

    Returns:
        bool: _description_
    """
    try:
        wav = wave.open(fname, 'rb')
    except wave.Error:
        print(f"error: failed to open '{fname}' as WAV file")
        return False

    if wav.getnchannels() != 1 and wav.getnchannels() != 2:
        print(f"WAV file '{fname}' must be mono or stereo")
        return False

    if stereo and wav.getnchannels() != 2:
        print(f"WAV file '{fname}' must be stereo for diarization")
        return False

    if wav.getframerate() != whisper_cpp.WHISPER_SAMPLE_RATE:
        print(f"WAV file '{fname}' must be {whisper_cpp.WHISPER_SAMPLE_RATE/1000} kHz")
        return False

    if wav.getsampwidth() != 2:
        print(f"WAV file '{fname}' must be 16-bit")
        return False

    n = wav.getnframes()

    pcm16 = np.frombuffer(wav.readframes(n), dtype=np.int16)

    wav.close()

    pcmf32.clear()
    pcmf32s.clear()

    if wav.getnchannels() == 1:
        pcmf32.extend(pcm16.astype(np.float32) / 32768.0)
    else:
        pcmf32.extend((pcm16[::2] + pcm16[1::2]).astype(np.float32) / 65536.0)

    if stereo:
        pcmf32s.extend([[], []])
        pcmf32s[0].extend(pcm16[::2].astype(np.float32) / 32768.0)
        pcmf32s[1].extend(pcm16[1::2].astype(np.float32) / 32768.0)

    return True

def to_timestamp(t: int, comma: bool = False) -> str:
    """Converts a timestamp to a string.
    500 -> 00:05.000
    6000 -> 01:00.000

    Args:
        t (int): timestamp
        comma (bool, optional): whether to add a comma after the timestamp. Defaults to False.

    Returns:
        str: timestamp as a string
    """
    msec = t * 10
    hour = math.floor(msec / (1000 * 60 * 60))
    msec = msec - hour * (1000 * 60 * 60)
    minute = math.floor(msec / (1000 * 60))
    msec = msec - minute * (1000 * 60)
    second = math.floor(msec / 1000)
    msec = msec - second * 1000

    sep = "," if comma else "."

    return f"{hour:02}:{minute:02}.{second:02}{sep}{msec:03}"

def to_lrc_timestamp(t: int) -> str:
    """Converts a timestamp to a lrc style string.
    500 -> 00:05.00
    6000 -> 01:00.00

    Args:
        t (int): timestamp
        comma (bool, optional): whether to add a comma after the timestamp. Defaults to False.

    Returns:
        str: timestamp as a string
    """
    msec = t * 10
    minute = math.floor(msec / (1000 * 60))
    msec = msec - minute * (1000 * 60)
    second = math.floor(msec / 1000)
    msec = msec - second * 1000

    return f"{minute:02}.{second:02}.{msec:03}"

def escape_double_quotes_and_backslashes(input_str: str) -> str:
    """Escapes double quotes and backslashes in a string.

    Args:
        s (str): string to escape

    Returns:
        str: escaped string
    """
    escaped = ""
    for _, char in enumerate(input_str):
        if char in ['"', "\\"]:
            escaped += "\\"
        escaped += char

    return escaped

def timestamp_to_sample(time: float, n_samples: int) -> int:
    return int(max(0, min(n_samples - 1, math.floor(time * whisper_cpp.WHISPER_SAMPLE_RATE)/100)))

def estimate_diarization_speaker(pcmf32s: list, time_start: int, time_stop: int, id_only: bool) -> str:
    n_samples = len(pcmf32s[0])

    is0: int = timestamp_to_sample(time_start, n_samples)
    is1: int = timestamp_to_sample(time_stop, n_samples)

    energy0: float = 0.0
    energy1: float = 0.0

    for i in range(is0, is1):
        energy0 += math.fabs(pcmf32s[0][i])
        energy1 += math.fabs(pcmf32s[1][i])

    if energy0 > 1.1*energy1:
        speaker="0"
    elif energy1 > 1.1*energy0:
        speaker="1"
    else:
        speaker="?"

    if not id_only:
        speaker = f"(speaker {speaker})"

    return speaker

def get_color(i: int) -> str:
    colors = [
        "\033[38;5;196m", "\033[38;5;202m", "\033[38;5;208m", "\033[38;5;214m", "\033[38;5;220m",
        "\033[38;5;226m", "\033[38;5;190m", "\033[38;5;154m", "\033[38;5;118m", "\033[38;5;82m",
    ]
    color_id = max(0, min(len(colors) - 1, int(math.pow(i, 3) * len(colors))))

    return colors[color_id]