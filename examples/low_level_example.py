from ctypes import (
    c_float,
)
from whisper_cpp.whisper_cpp import (
    whisper_init_from_file,
    whisper_full_default_params,
    whisper_full_parallel,
    whisper_full_n_segments,
    whisper_full_get_segment_text,
    WHISPER_SAMPLING_GREEDY,
)
from whisper_cpp.utils import read_wav

pcmf32 = []
pcmf32s = []

if not read_wav("samples/samples.wav", pcmf32, pcmf32s, False):
    raise RuntimeError("Failed to read WAV file!")

whisper_ctx = whisper_init_from_file("../models/ggml-large.bin".encode("utf-8"))

wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)

pcmf32_array = (c_float * len(pcmf32))(*(i for i in pcmf32))

whisper_full_parallel(
    ctx=whisper_ctx,
    params=wparams,
    samples=pcmf32_array,
    n_samples=len(pcmf32),
    n_processors=1
)

result = ""
n_segments = whisper_full_n_segments(whisper_ctx)
for i in range(n_segments):
    text = whisper_full_get_segment_text(whisper_ctx, i)
    result = result + text.decode('utf-8')

print(result)