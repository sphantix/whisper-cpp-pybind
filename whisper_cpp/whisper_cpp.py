import sys
import os
import ctypes
from ctypes import (
    c_int,
    c_float,
    c_char_p,
    c_int64,
    c_void_p,
    c_bool,
    POINTER,
    Structure,
    Array,
    c_uint8,
    c_size_t,
    CFUNCTYPE,
    CDLL,
)
import pathlib
from typing import List, Union


# Load the library
def _load_shared_library(lib_base_name: str):
    # Construct the paths to the possible shared library names
    _base_path = pathlib.Path(__file__).parent.resolve()
    # Searching for the library in the current directory under the name "libllama" (default name
    # for llamacpp) and "llama" (default name for this repo)
    _lib_paths: List[pathlib.Path] = []
    # Determine the file extension based on the platform
    if sys.platform.startswith("linux"):
        _lib_paths += [
            _base_path / f"lib{lib_base_name}.so",
        ]
    elif sys.platform == "darwin":
        _lib_paths += [
            _base_path / f"lib{lib_base_name}.so",
            _base_path / f"lib{lib_base_name}.dylib",
        ]
    elif sys.platform == "win32":
        _lib_paths += [
            _base_path / f"{lib_base_name}.dll",
        ]
    else:
        raise RuntimeError("Unsupported platform")

    if "WHISPER_CPP_LIB" in os.environ:
        lib_base_name = os.environ["WHISPER_CPP_LIB"]
        _lib = pathlib.Path(lib_base_name)
        _base_path = _lib.parent.resolve()
        _lib_paths = [_lib.resolve()]

    cdll_args = dict()  # type: ignore
    # Add the library directory to the DLL search path on Windows (if needed)
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        os.add_dll_directory(str(_base_path))
        if "CUDA_PATH" in os.environ:
            os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"], "bin"))
            os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"], "lib"))
        cdll_args["winmode"] = 0

    # Try to load the shared library, handling potential errors
    for _lib_path in _lib_paths:
        if _lib_path.exists():
            try:
                return CDLL(str(_lib_path), **cdll_args)
            except Exception as e:
                raise RuntimeError(f"Failed to load shared library '{_lib_path}': {e}")

    raise FileNotFoundError(
        f"Shared library with base name '{lib_base_name}' not found"
    )


# Specify the base name of the shared library to load
_LIB_BASE_NAME = "whisper"

# Load the library
_lib = _load_shared_library(_LIB_BASE_NAME)

# Misc
c_float_p = POINTER(c_float)
c_uint8_p = POINTER(c_uint8)
c_size_t_p = POINTER(c_size_t)

# whisper.h bindings

GGML_USE_CUBLAS = hasattr(_lib, "ggml_init_cublas")
GGML_CUDA_MAX_DEVICES = ctypes.c_int(16)

# define WHISPER_SAMPLE_RATE 16000
WHISPER_SAMPLE_RATE = 16000

# define WHISPER_N_FFT       400
WHISPER_N_FFT = ctypes.c_int(400)

# define WHISPER_N_MEL       80
WHISPER_N_MEL = ctypes.c_int(80)

# define WHISPER_HOP_LENGTH  160
WHISPER_HOP_LENGTH = ctypes.c_int(160)

# define WHISPER_CHUNK_SIZE  30
WHISPER_CHUNK_SIZE = ctypes.c_int(30)

# struct whisper_context;
whisper_context_p = c_void_p

# struct whisper_state;
whisper_state_p = c_void_p

# struct whisper_full_params;
whisper_full_params_p = c_void_p

# typedef int whisper_token;
whisper_token = ctypes.c_int
whisper_token_p = POINTER(whisper_token)

# typedef struct whisper_token_data {
#     whisper_token id;  // token id
#     whisper_token tid; // forced timestamp token id

#     float p;           // probability of the token
#     float plog;        // log probability of the token
#     float pt;          // probability of the timestamp token
#     float ptsum;       // sum of probabilities of all timestamp tokens

#     // token-level timestamp data
#     // do not use if you haven't computed token-level timestamps
#     int64_t t0;        // start time of the token
#     int64_t t1;        //   end time of the token

#     float vlen;        // voice length of the token
# } whisper_token_data;
class whisper_token_data(Structure):
    _fields_ = [
        ("id", whisper_token),
        ("tid", whisper_token),
        ("p", ctypes.c_float),
        ("plog", ctypes.c_float),
        ("pt", ctypes.c_float),
        ("ptsum", ctypes.c_float),
        ("t0", ctypes.c_int64),
        ("t1", ctypes.c_int64),
        ("vlen", ctypes.c_float),
    ]

whisper_token_data_p = POINTER(whisper_token_data)

# typedef struct whisper_model_loader {
#     void * context;

#     size_t (*read)(void * ctx, void * output, size_t read_size);
#     bool    (*eof)(void * ctx);
#     void  (*close)(void * ctx);
# } whisper_model_loader;
class whisper_model_loader(Structure):
    _fields_ = [
        ("context", c_void_p),
        ("read", c_void_p),
        ("eof", c_void_p),
        ("close", c_void_p),
    ]

whisper_model_loader_p = POINTER(whisper_model_loader)

# WHISPER_API struct whisper_context * whisper_init_from_file(const char * path_model);
def whisper_init_from_file(path_model: bytes) -> whisper_context_p:
    return _lib.whisper_init_from_file(path_model)

_lib.whisper_init_from_file.argtypes = [c_char_p]
_lib.whisper_init_from_file.restype = whisper_context_p

# WHISPER_API struct whisper_context * whisper_init_from_buffer(void * buffer, size_t buffer_size);
def whisper_init_from_buffer(buffer: bytes, buffer_size: c_size_t) -> whisper_context_p:
    return _lib.whisper_init_from_buffer(buffer, buffer_size)

_lib.whisper_init_from_buffer.argtypes = [c_void_p, c_size_t]
_lib.whisper_init_from_buffer.restype = whisper_context_p

# WHISPER_API struct whisper_context * whisper_init(struct whisper_model_loader * loader);
def whisper_init(loader: whisper_model_loader_p) -> whisper_context_p:
    return _lib.whisper_init(loader)

_lib.whisper_init.argtypes = [whisper_model_loader_p]
_lib.whisper_init.restype = whisper_context_p

# WHISPER_API struct whisper_context * whisper_init_from_file_no_state(const char * path_model);
def whisper_init_from_file_no_state(path_model: bytes) -> whisper_context_p:
    return _lib.whisper_init_from_file_no_state(path_model)

_lib.whisper_init_from_file_no_state.argtypes = [c_char_p]
_lib.whisper_init_from_file_no_state.restype = whisper_context_p

# WHISPER_API struct whisper_context * whisper_init_from_buffer_no_state(void * buffer, size_t buffer_size);
def whisper_init_from_buffer_no_state(
    buffer: bytes,
    buffer_size: c_size_t
) -> whisper_context_p:
    return _lib.whisper_init_from_buffer_no_state(buffer, buffer_size)

_lib.whisper_init_from_buffer_no_state.argtypes = [c_void_p, c_size_t]
_lib.whisper_init_from_buffer_no_state.restype = whisper_context_p

# WHISPER_API struct whisper_context * whisper_init_no_state(struct whisper_model_loader * loader);
def whisper_init_no_state(loader: whisper_model_loader_p) -> whisper_context_p:
    return _lib.whisper_init_no_state(loader)

_lib.whisper_init_no_state.argtypes = [whisper_model_loader_p]
_lib.whisper_init_no_state.restype = whisper_context_p

# WHISPER_API struct whisper_state * whisper_init_state(struct whisper_context * ctx);
def whisper_init_state(ctx: whisper_context_p) -> whisper_state_p:
    return _lib.whisper_init_state(ctx)

_lib.whisper_init_state.argtypes = [whisper_context_p]
_lib.whisper_init_state.restype = whisper_state_p

# // Given a context, enable use of OpenVINO for encode inference.
# // model_path: Optional path to OpenVINO encoder IR model. If set to nullptr,
# //                      the path will be generated from the ggml model path that was passed
# //                      in to whisper_init_from_file. For example, if 'path_model' was
# //                      "/path/to/ggml-base.en.bin", then OpenVINO IR model path will be
# //                      assumed to be "/path/to/ggml-base.en-encoder-openvino.xml".
# // device: OpenVINO device to run inference on ("CPU", "GPU", etc.)
# // cache_dir: Optional cache directory that can speed up init time, especially for
# //                     GPU, by caching compiled 'blobs' there.
# //                     Set to nullptr if not used.
# // Returns 0 on success. If OpenVINO is not enabled in build, this simply returns 1.
# WHISPER_API int whisper_ctx_init_openvino_encoder(
#     struct whisper_context * ctx,
#                 const char * model_path,
#                 const char * device,
#                 const char * cache_dir);
def whisper_ctx_init_openvino_encoder(
    ctx: whisper_context_p,
    model_path: bytes,
    device: bytes,
    cache_dir: bytes,
) -> int:
    return _lib.whisper_ctx_init_openvino_encoder(ctx, model_path, device, cache_dir)

_lib.whisper_ctx_init_openvino_encoder.argtypes = [whisper_context_p, c_char_p, c_char_p, c_char_p]
_lib.whisper_ctx_init_openvino_encoder.restype = c_int

# // Frees all allocated memory
# WHISPER_API void whisper_free      (struct whisper_context * ctx);
def whisper_free(ctx: whisper_context_p) -> None:
    return _lib.whisper_free(ctx)

_lib.whisper_free.argtypes = [whisper_context_p]
_lib.whisper_free.restype = None

# WHISPER_API void whisper_free_state(struct whisper_state * state)
def whisper_free_state(state: whisper_state_p) -> None:
    return _lib.whisper_free_state(state)

_lib.whisper_free_state.argtypes = [whisper_state_p]
_lib.whisper_free_state.restype = None

# WHISPER_API void whisper_free_params(struct whisper_full_params * params);
def whisper_free_params(params: whisper_full_params_p) -> None:
    return _lib.whisper_free_params(params)

_lib.whisper_free_params.argtypes = [whisper_full_params_p]
_lib.whisper_free_params.restype = None

# // Convert RAW PCM audio to log mel spectrogram.
# // The resulting spectrogram is stored inside the default state of the provided whisper context.
# // Returns 0 on success
# WHISPER_API int whisper_pcm_to_mel(
#         struct whisper_context * ctx,
#                    const float * samples,
#                            int   n_samples,
#                            int   n_threads);
def whisper_pcm_to_mel(
    ctx: whisper_context_p,
    samples: Array[c_float],
    n_samples: int,
    n_threads: int,
) -> int:
    return _lib.whisper_pcm_to_mel(ctx, samples, n_samples, n_threads)

_lib.whisper_pcm_to_mel.argtypes = [whisper_context_p, c_float_p, c_int, c_int]
_lib.whisper_pcm_to_mel.restype = c_int

# WHISPER_API int whisper_pcm_to_mel_with_state(
#         struct whisper_context * ctx,
#           struct whisper_state * state,
#                    const float * samples,
#                            int   n_samples,
#                            int   n_threads);
def whisper_pcm_to_mel_with_state(
    ctx: whisper_context_p,
    state: whisper_state_p,
    samples: Array[c_float],
    n_samples: int,
    n_threads: int,
) -> int:
    return _lib.whisper_pcm_to_mel_with_state(ctx, state, samples, n_samples, n_threads)

_lib.whisper_pcm_to_mel_with_state.argtypes = [
    whisper_context_p,
    whisper_state_p,
    c_float_p,
    c_int,
    c_int
]
_lib.whisper_pcm_to_mel_with_state.restype = c_int

# // Convert RAW PCM audio to log mel spectrogram but applies a Phase Vocoder to speed up the audio x2.
# // The resulting spectrogram is stored inside the default state of the provided whisper context.
# // Returns 0 on success
# WHISPER_API int whisper_pcm_to_mel_phase_vocoder(
#     struct whisper_context * ctx,
#                const float * samples,
#                        int   n_samples,
#                        int   n_threads);
def whisper_pcm_to_mel_phase_vocoder(
    ctx: whisper_context_p,
    samples: Array[c_float],
    n_samples: int,
    n_threads: int,
) -> int:
    return _lib.whisper_pcm_to_mel_phase_vocoder(ctx, samples, n_samples, n_threads)

_lib.whisper_pcm_to_mel_phase_vocoder.argtypes = [whisper_context_p, c_float_p, c_int, c_int]
_lib.whisper_pcm_to_mel_phase_vocoder.restype = c_int

# WHISPER_API int whisper_pcm_to_mel_phase_vocoder_with_state(
#     struct whisper_context * ctx,
#       struct whisper_state * state,
#                const float * samples,
#                        int   n_samples,
#                        int   n_threads);
def whisper_pcm_to_mel_phase_vocoder_with_state(
    ctx: whisper_context_p,
    state: whisper_state_p,
    samples: Array[c_float],
    n_samples: int,
    n_threads: int,
) -> int:
    return _lib.whisper_pcm_to_mel_phase_vocoder_with_state(
        ctx,
        state,
        samples,
        n_samples,
        n_threads
    )

_lib.whisper_pcm_to_mel_phase_vocoder_with_state.argtypes = [
    whisper_context_p,
    whisper_state_p,
    c_float_p,
    c_int,
    c_int
]
_lib.whisper_pcm_to_mel_phase_vocoder_with_state.restype = c_int

# // This can be used to set a custom log mel spectrogram inside the default state of the provided whisper context.
# // Use this instead of whisper_pcm_to_mel() if you want to provide your own log mel spectrogram.
# // n_mel must be 80
# // Returns 0 on success
# WHISPER_API int whisper_set_mel(
#         struct whisper_context * ctx,
#                    const float * data,
#                            int   n_len,
#                            int   n_mel);
def whisper_set_mel(
    ctx: whisper_context_p,
    data: Array[c_float],
    n_len: int,
    n_mel: int,
) -> int:
    return _lib.whisper_set_mel(ctx, data, n_len, n_mel)

_lib.whisper_set_mel.argtypes = [whisper_context_p, c_float_p, c_int, c_int]
_lib.whisper_set_mel.restype = c_int

# WHISPER_API int whisper_set_mel_with_state(
#         struct whisper_context * ctx,
#           struct whisper_state * state,
#                    const float * data,
#                            int   n_len,
#                            int   n_mel);
def whisper_set_mel_with_state(
    ctx: whisper_context_p,
    state: whisper_state_p,
    data: Array[c_float],
    n_len: int,
    n_mel: int,
) -> int:
    return _lib.whisper_set_mel_with_state(ctx, state, data, n_len, n_mel)

_lib.whisper_set_mel_with_state.argtypes = [
    whisper_context_p,
    whisper_state_p,
    c_float_p,
    c_int,
    c_int
]
_lib.whisper_set_mel_with_state.restype = c_int

# // Run the Whisper encoder on the log mel spectrogram stored inside the default state in the provided whisper context.
# // Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first.
# // offset can be used to specify the offset of the first frame in the spectrogram.
# // Returns 0 on success
# WHISPER_API int whisper_encode(
#         struct whisper_context * ctx,
#                            int   offset,
#                            int   n_threads);
def whisper_encode(
    ctx: whisper_context_p,
    offset: int,
    n_threads: int,
) -> int:
    return _lib.whisper_encode(ctx, offset, n_threads)

_lib.whisper_encode.argtypes = [whisper_context_p, c_int, c_int]
_lib.whisper_encode.restype = c_int

# WHISPER_API int whisper_encode_with_state(
#         struct whisper_context * ctx,
#           struct whisper_state * state,
#                            int   offset,
#                            int   n_threads);
def whisper_encode_with_state(
    ctx: whisper_context_p,
    state: whisper_state_p,
    offset: int,
    n_threads: int,
) -> int:
    return _lib.whisper_encode_with_state(ctx, state, offset, n_threads)

_lib.whisper_encode_with_state.argtypes = [whisper_context_p, whisper_state_p, c_int, c_int]
_lib.whisper_encode_with_state.restype = c_int

# // Run the Whisper decoder to obtain the logits and probabilities for the next token.
# // Make sure to call whisper_encode() first.
# // tokens + n_tokens is the provided context for the decoder.
# // n_past is the number of tokens to use from previous decoder calls.
# // Returns 0 on success
# // TODO: add support for multiple decoders
# WHISPER_API int whisper_decode(
#         struct whisper_context * ctx,
#            const whisper_token * tokens,
#                            int   n_tokens,
#                            int   n_past,
#                            int   n_threads);
def whisper_decode(
    ctx: whisper_context_p,
    tokens: whisper_token_p,
    n_tokens: int,
    n_past: int,
    n_threads: int,
) -> int:
    return _lib.whisper_decode(ctx, tokens, n_tokens, n_past, n_threads)

_lib.whisper_decode.argtypes = [whisper_context_p, whisper_token_p, c_int, c_int, c_int]
_lib.whisper_decode.restype = c_int

# WHISPER_API int whisper_decode_with_state(
#         struct whisper_context * ctx,
#           struct whisper_state * state,
#            const whisper_token * tokens,
#                            int   n_tokens,
#                            int   n_past,
#                            int   n_threads);
def whisper_decode_with_state(
    ctx: whisper_context_p,
    state: whisper_state_p,
    tokens: whisper_token_p,
    n_tokens: int,
    n_past: int,
    n_threads: int,
) -> int:
    return _lib.whisper_decode_with_state(ctx, state, tokens, n_tokens, n_past, n_threads)

_lib.whisper_decode_with_state.argtypes = [whisper_context_p,
                                           whisper_state_p,
                                           whisper_token_p,
                                           c_int,
                                           c_int,
                                           c_int]
_lib.whisper_decode_with_state.restype = c_int

# // Convert the provided text into tokens.
# // The tokens pointer must be large enough to hold the resulting tokens.
# // Returns the number of tokens on success, no more than n_max_tokens
# // Returns -1 on failure
# // TODO: not sure if correct
# WHISPER_API int whisper_tokenize(
#         struct whisper_context * ctx,
#                     const char * text,
#                  whisper_token * tokens,
#                            int   n_max_tokens);
def whisper_tokenize(
    ctx: whisper_context_p,
    text: bytes,
    tokens: whisper_token_p,
    n_max_tokens: int,
) -> int:
    return _lib.whisper_tokenize(ctx, text, tokens, n_max_tokens)

_lib.whisper_tokenize.argtypes = [whisper_context_p, c_char_p, whisper_token_p, c_int]
_lib.whisper_tokenize.restype = c_int

# // Largest language id (i.e. number of available languages - 1)
# WHISPER_API int whisper_lang_max_id();
def whisper_lang_max_id() -> int:
    return _lib.whisper_lang_max_id()

_lib.whisper_lang_max_id.argtypes = []
_lib.whisper_lang_max_id.restype = c_int

# // Return the id of the specified language, returns -1 if not found
# // Examples:
# //   "de" -> 2
# //   "german" -> 2
# WHISPER_API int whisper_lang_id(const char * lang);
def whisper_lang_id(lang: bytes) -> int:
    return _lib.whisper_lang_id(lang)

_lib.whisper_lang_id.argtypes = [c_char_p]
_lib.whisper_lang_id.restype = c_int

# // Return the short string of the specified language id (e.g. 2 -> "de"), returns nullptr if not found
# WHISPER_API const char * whisper_lang_str(int id);
def whisper_lang_str(id: int) -> bytes:
    return _lib.whisper_lang_str(id)

_lib.whisper_lang_str.argtypes = [c_int]
_lib.whisper_lang_str.restype = c_char_p

# // Use mel data at offset_ms to try and auto-detect the spoken language
# // Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first
# // Returns the top language id or negative on failure
# // If not null, fills the lang_probs array with the probabilities of all languages
# // The array must be whisper_lang_max_id() + 1 in size
# // ref: https://github.com/openai/whisper/blob/main/whisper/decoding.py#L18-L69
# WHISPER_API int whisper_lang_auto_detect(
#         struct whisper_context * ctx,
#                            int   offset_ms,
#                            int   n_threads,
#                          float * lang_probs);
def whisper_lang_auto_detect(
    ctx: whisper_context_p,
    offset_ms: int,
    n_threads: int,
    lang_probs: Array[c_float],
) -> int:
    return _lib.whisper_lang_auto_detect(ctx, offset_ms, n_threads, lang_probs)

_lib.whisper_lang_auto_detect.argtypes = [whisper_context_p, c_int, c_int, c_float_p]
_lib.whisper_lang_auto_detect.restype = c_int

# WHISPER_API int whisper_lang_auto_detect_with_state(
#         struct whisper_context * ctx,
#           struct whisper_state * state,
#                            int   offset_ms,
#                            int   n_threads,
#                          float * lang_probs);
def whisper_lang_auto_detect_with_state(
    ctx: whisper_context_p,
    state: whisper_state_p,
    offset_ms: int,
    n_threads: int,
    lang_probs: Array[c_float],
) -> int:
    return _lib.whisper_lang_auto_detect_with_state(ctx, state, offset_ms, n_threads, lang_probs)

_lib.whisper_lang_auto_detect_with_state.argtypes = [
    whisper_context_p,
    whisper_state_p,
    c_int,
    c_int,
    c_float_p
]
_lib.whisper_lang_auto_detect_with_state.restype = c_int

# WHISPER_API int whisper_n_len           (struct whisper_context * ctx); // mel length
def whisper_n_len(ctx: whisper_context_p) -> int:
    return _lib.whisper_n_len(ctx)

_lib.whisper_n_len.argtypes = [whisper_context_p]
_lib.whisper_n_len.restype = c_int

# WHISPER_API int whisper_n_len_from_state(struct whisper_state * state); // mel length
def whisper_n_len_from_state(state: whisper_state_p) -> int:
    return _lib.whisper_n_len_from_state(state)

_lib.whisper_n_len_from_state.argtypes = [whisper_state_p]
_lib.whisper_n_len_from_state.restype = c_int

# WHISPER_API int whisper_n_vocab         (struct whisper_context * ctx);
def whisper_n_vocab(ctx: whisper_context_p) -> int:
    return _lib.whisper_n_vocab(ctx)

_lib.whisper_n_vocab.argtypes = [whisper_context_p]
_lib.whisper_n_vocab.restype = c_int

# WHISPER_API int whisper_n_text_ctx      (struct whisper_context * ctx);
def whisper_n_text_ctx(ctx: whisper_context_p) -> int:
    return _lib.whisper_n_text_ctx(ctx)

_lib.whisper_n_text_ctx.argtypes = [whisper_context_p]
_lib.whisper_n_text_ctx.restype = c_int

# WHISPER_API int whisper_n_audio_ctx     (struct whisper_context * ctx);
def whisper_n_audio_ctx(ctx: whisper_context_p) -> int:
    return _lib.whisper_n_audio_ctx(ctx)

_lib.whisper_n_audio_ctx.argtypes = [whisper_context_p]
_lib.whisper_n_audio_ctx.restype = c_int

# WHISPER_API int whisper_is_multilingual (struct whisper_context * ctx);
def whisper_is_multilingual(ctx: whisper_context_p) -> int:
    return _lib.whisper_is_multilingual(ctx)

_lib.whisper_is_multilingual.argtypes = [whisper_context_p]
_lib.whisper_is_multilingual.restype = c_int

# WHISPER_API int whisper_model_n_vocab      (struct whisper_context * ctx);
def whisper_model_n_vocab(ctx: whisper_context_p) -> int:
    return _lib.whisper_model_n_vocab(ctx)

_lib.whisper_model_n_vocab.argtypes = [whisper_context_p]
_lib.whisper_model_n_vocab.restype = c_int

# WHISPER_API int whisper_model_n_audio_ctx  (struct whisper_context * ctx);
def whisper_model_n_audio_ctx(ctx: whisper_context_p) -> int:
    return _lib.whisper_model_n_audio_ctx(ctx)

_lib.whisper_model_n_audio_ctx.argtypes = [whisper_context_p]
_lib.whisper_model_n_audio_ctx.restype = c_int

# WHISPER_API int whisper_model_n_audio_state(struct whisper_context * ctx);
def whisper_model_n_audio_state(ctx: whisper_context_p) -> int:
    return _lib.whisper_model_n_audio_state(ctx)

_lib.whisper_model_n_audio_state.argtypes = [whisper_context_p]
_lib.whisper_model_n_audio_state.restype = c_int

# WHISPER_API int whisper_model_n_audio_head (struct whisper_context * ctx);
def whisper_model_n_audio_head(ctx: whisper_context_p) -> int:
    return _lib.whisper_model_n_audio_head(ctx)

_lib.whisper_model_n_audio_head.argtypes = [whisper_context_p]
_lib.whisper_model_n_audio_head.restype = c_int

# WHISPER_API int whisper_model_n_audio_layer(struct whisper_context * ctx);
def whisper_model_n_audio_layer(ctx: whisper_context_p) -> int:
    return _lib.whisper_model_n_audio_layer(ctx)

_lib.whisper_model_n_audio_layer.argtypes = [whisper_context_p]
_lib.whisper_model_n_audio_layer.restype = c_int

# WHISPER_API int whisper_model_n_text_ctx   (struct whisper_context * ctx);
def whisper_model_n_text_ctx(ctx: whisper_context_p) -> int:
    return _lib.whisper_model_n_text_ctx(ctx)

_lib.whisper_model_n_text_ctx.argtypes = [whisper_context_p]
_lib.whisper_model_n_text_ctx.restype = c_int

# WHISPER_API int whisper_model_n_text_state (struct whisper_context * ctx);
def whisper_model_n_text_state(ctx: whisper_context_p) -> int:
    return _lib.whisper_model_n_text_state(ctx)

_lib.whisper_model_n_text_state.argtypes = [whisper_context_p]
_lib.whisper_model_n_text_state.restype = c_int

# WHISPER_API int whisper_model_n_text_head  (struct whisper_context * ctx);
def whisper_model_n_text_head(ctx: whisper_context_p) -> int:
    return _lib.whisper_model_n_text_head(ctx)

_lib.whisper_model_n_text_head.argtypes = [whisper_context_p]
_lib.whisper_model_n_text_head.restype = c_int

# WHISPER_API int whisper_model_n_text_layer (struct whisper_context * ctx);
def whisper_model_n_text_layer(ctx: whisper_context_p) -> int:
    return _lib.whisper_model_n_text_layer(ctx)

_lib.whisper_model_n_text_layer.argtypes = [whisper_context_p]
_lib.whisper_model_n_text_layer.restype = c_int

# WHISPER_API int whisper_model_n_mels       (struct whisper_context * ctx);
def whisper_model_n_mels(ctx: whisper_context_p) -> int:
    return _lib.whisper_model_n_mels(ctx)

_lib.whisper_model_n_mels.argtypes = [whisper_context_p]
_lib.whisper_model_n_mels.restype = c_int

# WHISPER_API int whisper_model_ftype        (struct whisper_context * ctx);
def whisper_model_ftype(ctx: whisper_context_p) -> int:
    return _lib.whisper_model_ftype(ctx)

_lib.whisper_model_ftype.argtypes = [whisper_context_p]
_lib.whisper_model_ftype.restype = c_int

# WHISPER_API int whisper_model_type         (struct whisper_context * ctx);
def whisper_model_type(ctx: whisper_context_p) -> int:
    return _lib.whisper_model_type(ctx)

_lib.whisper_model_type.argtypes = [whisper_context_p]
_lib.whisper_model_type.restype = c_int

# // Token logits obtained from the last call to whisper_decode()
# // The logits for the last token are stored in the last row
# // Rows: n_tokens
# // Cols: n_vocab
# WHISPER_API float * whisper_get_logits           (struct whisper_context * ctx);
def whisper_get_logits(ctx: whisper_context_p) -> Array[c_float]:
    return _lib.whisper_get_logits(ctx)

_lib.whisper_get_logits.argtypes = [whisper_context_p]
_lib.whisper_get_logits.restype = c_float_p

# WHISPER_API float * whisper_get_logits_from_state(struct whisper_state * state);
def whisper_get_logits_from_state(state: whisper_state_p) -> Array[c_float]:
    return _lib.whisper_get_logits_from_state(state)

_lib.whisper_get_logits_from_state.argtypes = [whisper_state_p]
_lib.whisper_get_logits_from_state.restype = c_float_p

# // Token Id -> String. Uses the vocabulary in the provided context
# WHISPER_API const char * whisper_token_to_str(struct whisper_context * ctx, whisper_token token);
def whisper_token_to_str(ctx: whisper_context_p, token: whisper_token) -> bytes:
    return _lib.whisper_token_to_str(ctx, token)

_lib.whisper_token_to_str.argtypes = [whisper_context_p, whisper_token]
_lib.whisper_token_to_str.restype = c_char_p

# WHISPER_API const char * whisper_model_type_readable(struct whisper_context * ctx);
def whisper_model_type_readable(ctx: whisper_context_p) -> bytes:
    return _lib.whisper_model_type_readable(ctx)

_lib.whisper_model_type_readable.argtypes = [whisper_context_p]
_lib.whisper_model_type_readable.restype = c_char_p


# // Special tokens
# WHISPER_API whisper_token whisper_token_eot (struct whisper_context * ctx);
def whisper_token_eot(ctx: whisper_context_p) -> whisper_token:
    return _lib.whisper_token_eot(ctx)

_lib.whisper_token_eot.argtypes = [whisper_context_p]
_lib.whisper_token_eot.restype = whisper_token

# WHISPER_API whisper_token whisper_token_sot (struct whisper_context * ctx);
def whisper_token_sot(ctx: whisper_context_p) -> whisper_token:
    return _lib.whisper_token_sot(ctx)

_lib.whisper_token_sot.argtypes = [whisper_context_p]
_lib.whisper_token_sot.restype = whisper_token

# WHISPER_API whisper_token whisper_token_solm(struct whisper_context * ctx);
def whisper_token_solm(ctx: whisper_context_p) -> whisper_token:
    return _lib.whisper_token_solm(ctx)

_lib.whisper_token_solm.argtypes = [whisper_context_p]
_lib.whisper_token_solm.restype = whisper_token

# WHISPER_API whisper_token whisper_token_prev(struct whisper_context * ctx);
def whisper_token_prev(ctx: whisper_context_p) -> whisper_token:
    return _lib.whisper_token_prev(ctx)

_lib.whisper_token_prev.argtypes = [whisper_context_p]
_lib.whisper_token_prev.restype = whisper_token

# WHISPER_API whisper_token whisper_token_nosp(struct whisper_context * ctx);
def whisper_token_nosp(ctx: whisper_context_p) -> whisper_token:
    return _lib.whisper_token_nosp(ctx)

_lib.whisper_token_nosp.argtypes = [whisper_context_p]
_lib.whisper_token_nosp.restype = whisper_token

# WHISPER_API whisper_token whisper_token_not (struct whisper_context * ctx);
def whisper_token_not(ctx: whisper_context_p) -> whisper_token:
    return _lib.whisper_token_not(ctx)

_lib.whisper_token_not.argtypes = [whisper_context_p]
_lib.whisper_token_not.restype = whisper_token

# WHISPER_API whisper_token whisper_token_beg (struct whisper_context * ctx);
def whisper_token_beg(ctx: whisper_context_p) -> whisper_token:
    return _lib.whisper_token_beg(ctx)

_lib.whisper_token_beg.argtypes = [whisper_context_p]
_lib.whisper_token_beg.restype = whisper_token

# WHISPER_API whisper_token whisper_token_lang(struct whisper_context * ctx, int lang_id);
def whisper_token_lang(ctx: whisper_context_p, lang_id: int) -> whisper_token:
    return _lib.whisper_token_lang(ctx, lang_id)

_lib.whisper_token_lang.argtypes = [whisper_context_p, c_int]
_lib.whisper_token_lang.restype = whisper_token

# // Task tokens
# WHISPER_API whisper_token whisper_token_translate (struct whisper_context * ctx);
def whisper_token_translate(ctx: whisper_context_p) -> whisper_token:
    return _lib.whisper_token_translate(ctx)

_lib.whisper_token_translate.argtypes = [whisper_context_p]
_lib.whisper_token_translate.restype = whisper_token

# WHISPER_API whisper_token whisper_token_transcribe(struct whisper_context * ctx);
def whisper_token_transcribe(ctx: whisper_context_p) -> whisper_token:
    return _lib.whisper_token_transcribe(ctx)

_lib.whisper_token_transcribe.argtypes = [whisper_context_p]
_lib.whisper_token_transcribe.restype = whisper_token

# // Performance information from the default state.
# WHISPER_API void whisper_print_timings(struct whisper_context * ctx);
def whisper_print_timings(ctx: whisper_context_p) -> None:
    return _lib.whisper_print_timings(ctx)

_lib.whisper_print_timings.argtypes = [whisper_context_p]
_lib.whisper_print_timings.restype = None

# WHISPER_API void whisper_reset_timings(struct whisper_context * ctx);
def whisper_reset_timings(ctx: whisper_context_p) -> None:
    return _lib.whisper_reset_timings(ctx)

_lib.whisper_reset_timings.argtypes = [whisper_context_p]
_lib.whisper_reset_timings.restype = None

# // Print system information
# WHISPER_API const char * whisper_print_system_info(void);
def whisper_print_system_info() -> bytes:
    return _lib.whisper_print_system_info()

_lib.whisper_print_system_info.argtypes = []
_lib.whisper_print_system_info.restype = c_char_p

# ////////////////////////////////////////////////////////////////////////////

# // Available sampling strategies
# enum whisper_sampling_strategy {
#     WHISPER_SAMPLING_GREEDY,      // similar to OpenAI's GreedyDecoder
#     WHISPER_SAMPLING_BEAM_SEARCH, // similar to OpenAI's BeamSearchDecoder
# };
WHISPER_SAMPLING_GREEDY = c_int(0)
WHISPER_SAMPLING_BEAM_SEARCH = c_int(1)

class whisper_segment_callback_user_data(Structure):
    _fields_ = [
        ("no_timestamps", c_bool),
        ("print_colors", c_bool),
        ("print_special", c_bool),
        ("tinydiarize", c_bool),
        ("tdrz_speaker_turn", c_char_p),
]
# // Text segment callback
# // Called on every newly generated text segment
# // Use the whisper_full_...() functions to obtain the text segments
# typedef void (*whisper_new_segment_callback)(struct whisper_context * ctx, struct whisper_state * state, int n_new, void * user_data);
whisper_new_segment_callback_fn_t=CFUNCTYPE(None,
                                            whisper_context_p,
                                            whisper_state_p,
                                            c_int,
                                            c_void_p)

class whisper_progress_callback_user_data(Structure):
    _fields_ = [
        ("progress_step", c_int),
        ("progress_prev", c_int),
]
# // Progress callback
# typedef void (*whisper_progress_callback)(struct whisper_context * ctx, struct whisper_state * state, int progress, void * user_data);
whisper_progress_callback_fn_t=CFUNCTYPE(None,
                                         whisper_context_p,
                                         whisper_state_p,
                                         c_int,
                                         c_void_p)

class whisper_encoder_begin_callback_user_data(Structure):
    _fields_ = [
        ("is_aborted", c_bool),
]
# // Encoder begin callback
# // If not NULL, called before the encoder starts
# // If it returns false, the computation is aborted
# typedef bool (*whisper_encoder_begin_callback)(struct whisper_context * ctx, struct whisper_state * state, void * user_data);
whisper_encoder_begin_callback_fn_t=CFUNCTYPE(c_bool,
                                              whisper_context_p,
                                              whisper_state_p,
                                              c_void_p)

# // Logits filter callback
# // Can be used to modify the logits before sampling
# // If not NULL, called after applying temperature to logits
# typedef void (*whisper_logits_filter_callback)(
#         struct whisper_context * ctx,
#           struct whisper_state * state,
#       const whisper_token_data * tokens,
#                            int   n_tokens,
#                          float * logits,
#                           void * user_data);
whisper_logits_filter_callback_fn_t=CFUNCTYPE(None,
                                              whisper_context_p,
                                              whisper_state_p,
                                              whisper_token_data_p,
                                              c_int,
                                              c_float_p,
                                              c_void_p)

# // Parameters for the whisper_full() function
# // If you change the order or add new parameters, make sure to update the default values in whisper.cpp:
# // whisper_full_default_params()
# struct whisper_full_params {
#     enum whisper_sampling_strategy strategy;

#     int n_threads;
#     int n_max_text_ctx;     // max tokens to use from past text as prompt for the decoder
#     int offset_ms;          // start offset in ms
#     int duration_ms;        // audio duration to process in ms

#     bool translate;
#     bool no_context;        // do not use past transcription (if any) as initial prompt for the decoder
#     bool single_segment;    // force single segment output (useful for streaming)
#     bool print_special;     // print special tokens (e.g. <SOT>, <EOT>, <BEG>, etc.)
#     bool print_progress;    // print progress information
#     bool print_realtime;    // print results from within whisper.cpp (avoid it, use callback instead)
#     bool print_timestamps;  // print timestamps for each text segment when printing realtime

#     // [EXPERIMENTAL] token-level timestamps
#     bool  token_timestamps; // enable token-level timestamps
#     float thold_pt;         // timestamp token probability threshold (~0.01)
#     float thold_ptsum;      // timestamp token sum probability threshold (~0.01)
#     int   max_len;          // max segment length in characters
#     bool  split_on_word;    // split on word rather than on token (when used with max_len)
#     int   max_tokens;       // max tokens per segment (0 = no limit)

#     // [EXPERIMENTAL] speed-up techniques
#     // note: these can significantly reduce the quality of the output
#     bool speed_up;          // speed-up the audio by 2x using Phase Vocoder
#     bool debug_mode;        // enable debug_mode provides extra info (eg. Dump log_mel)
#     int  audio_ctx;         // overwrite the audio context size (0 = use default)

#     // [EXPERIMENTAL] [TDRZ] tinydiarize
#     bool tdrz_enable;       // enable tinydiarize speaker turn detection

#     // tokens to provide to the whisper decoder as initial prompt
#     // these are prepended to any existing text context from a previous call
#     const char * initial_prompt;
#     const whisper_token * prompt_tokens;
#     int prompt_n_tokens;

#     // for auto-detection, set to nullptr, "" or "auto"
#     const char * language;
#     bool detect_language;

#     // common decoding parameters:
#     bool suppress_blank;    // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L89
#     bool suppress_non_speech_tokens; // ref: https://github.com/openai/whisper/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/whisper/tokenizer.py#L224-L253

#     float temperature;      // initial decoding temperature, ref: https://ai.stackexchange.com/a/32478
#     float max_initial_ts;   // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L97
#     float length_penalty;   // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L267

#     // fallback parameters
#     // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L274-L278
#     float temperature_inc;
#     float entropy_thold;    // similar to OpenAI's "compression_ratio_threshold"
#     float logprob_thold;
#     float no_speech_thold;  // TODO: not implemented

#     struct {
#         int best_of;    // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L264
#     } greedy;

#     struct {
#         int beam_size;  // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L265

#         float patience; // TODO: not implemented, ref: https://arxiv.org/pdf/2204.05424.pdf
#     } beam_search;

#     // called for every newly generated text segment
#     whisper_new_segment_callback new_segment_callback;
#     void * new_segment_callback_user_data;

#     // called on each progress update
#     whisper_progress_callback progress_callback;
#     void * progress_callback_user_data;

#     // called each time before the encoder starts
#     whisper_encoder_begin_callback encoder_begin_callback;
#     void * encoder_begin_callback_user_data;

#     // called by each decoder to filter obtained logits
#     whisper_logits_filter_callback logits_filter_callback;
#     void * logits_filter_callback_user_data;
# };
class greedy(Structure):
    _fields_ = [
        ("best_of", c_int),
    ]

class beam_search(Structure):
    _fields_ = [
        ("beam_size", c_int),
        ("patience", c_float),
    ]

class whisper_full_params(Structure):
    _fields_ = [
        ("strategy", c_int),

        ("n_threads", c_int),
        ("n_max_text_ctx", c_int),
        ("offset_ms", c_int),
        ("duration_ms", c_int),

        ("translate", c_bool),
        ("no_context", c_bool),
        ("single_segment", c_bool),
        ("print_special", c_bool),
        ("print_progress", c_bool),
        ("print_realtime", c_bool),
        ("print_timestamps", c_bool),

        ("token_timestamps", c_bool),
        ("thold_pt", c_float),
        ("thold_ptsum", c_float),
        ("max_len", c_int),
        ("split_on_word", c_bool),
        ("max_tokens", c_int),

        ("speed_up", c_bool),
        ("debug_mode", c_bool),
        ("audio_ctx", c_int),

        ("tdrz_enable", c_bool),

        ("initial_prompt", c_char_p),
        ("prompt_tokens", whisper_token_p),
        ("prompt_n_tokens", c_int),

        ("language", c_char_p),
        ("detect_language", c_bool),

        ("suppress_blank", c_bool),
        ("suppress_non_speech_tokens", c_bool),

        ("temperature", c_float),
        ("max_initial_ts", c_float),
        ("length_penalty", c_float),

        ("temperature_inc", c_float),
        ("entropy_thold", c_float),
        ("logprob_thold", c_float),
        ("no_speech_thold", c_float),

        ("greedy", greedy),
        ("beam_search", beam_search),

        ("new_segment_callback", whisper_new_segment_callback_fn_t),
        ("new_segment_callback_user_data", c_void_p),

        ("progress_callback", whisper_progress_callback_fn_t),
        ("progress_callback_user_data", c_void_p),

        ("encoder_begin_callback", whisper_encoder_begin_callback_fn_t),
        ("encoder_begin_callback_user_data", c_void_p),

        ("logits_filter_callback", whisper_logits_filter_callback_fn_t),
        ("logits_filter_callback_user_data", c_void_p),
    ]

whisper_full_params_p = POINTER(whisper_full_params)

# // NOTE: this function allocates memory, and it is the responsibility of the caller to free the pointer - see whisper_free_params()
# WHISPER_API struct whisper_full_params * whisper_full_default_params_by_ref(enum whisper_sampling_strategy strategy);
def whisper_full_default_params_by_ref(strategy: int) -> whisper_full_params_p:
    return _lib.whisper_full_default_params_by_ref(strategy)

_lib.whisper_full_default_params_by_ref.argtypes = [c_int]
_lib.whisper_full_default_params_by_ref.restype = whisper_full_params_p

# WHISPER_API struct whisper_full_params whisper_full_default_params(enum whisper_sampling_strategy strategy);
def whisper_full_default_params(strategy: int) -> whisper_full_params:
    return _lib.whisper_full_default_params(strategy)

_lib.whisper_full_default_params.argtypes = [c_int]
_lib.whisper_full_default_params.restype = whisper_full_params

# // Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text
# // Not thread safe for same context
# // Uses the specified decoding strategy to obtain the text.
# WHISPER_API int whisper_full(
#             struct whisper_context * ctx,
#         struct whisper_full_params   params,
#                        const float * samples,
#                                int   n_samples);
def whisper_full(
    ctx: whisper_context_p,
    params: whisper_full_params,
    samples: Array[c_float],
    n_samples: int
) -> int:
    return _lib.whisper_full(ctx, params, samples, n_samples)

_lib.whisper_full.argtypes = [whisper_context_p, whisper_full_params, c_float_p, c_int]
_lib.whisper_full.restype = c_int

# WHISPER_API int whisper_full_with_state(
#             struct whisper_context * ctx,
#               struct whisper_state * state,
#         struct whisper_full_params   params,
#                        const float * samples,
#                                int   n_samples);
def whisper_full_with_state(
    ctx: whisper_context_p,
    state: whisper_state_p,
    params: whisper_full_params,
    samples: Array[c_float],
    n_samples: int
) -> int:
    return _lib.whisper_full_with_state(ctx, state, params, samples, n_samples)

_lib.whisper_full_with_state.argtypes = [whisper_context_p, whisper_state_p, whisper_full_params, c_float_p, c_int]
_lib.whisper_full_with_state.restype = c_int

# // Split the input audio in chunks and process each chunk separately using whisper_full_with_state()
# // Result is stored in the default state of the context
# // Not thread safe if executed in parallel on the same context.
# // It seems this approach can offer some speedup in some cases.
# // However, the transcription accuracy can be worse at the beginning and end of each chunk.
# WHISPER_API int whisper_full_parallel(
#             struct whisper_context * ctx,
#         struct whisper_full_params   params,
#                        const float * samples,
#                                int   n_samples,
#                                int   n_processors);
def whisper_full_parallel(
    ctx: whisper_context_p,
    params: whisper_full_params,
    samples: Array[c_float],
    n_samples: int,
    n_processors: int
) -> int:
    return _lib.whisper_full_parallel(ctx, params, samples, n_samples, n_processors)

_lib.whisper_full_parallel.argtypes = [
    whisper_context_p,
    whisper_full_params,
    c_float_p,
    c_int,
    c_int
]
_lib.whisper_full_parallel.restype = c_int

# // Number of generated text segments
# // A segment can be a few words, a sentence, or even a paragraph.
# WHISPER_API int whisper_full_n_segments           (struct whisper_context * ctx);
def whisper_full_n_segments(ctx: whisper_context_p) -> int:
    return _lib.whisper_full_n_segments(ctx)

_lib.whisper_full_n_segments.argtypes = [whisper_context_p]
_lib.whisper_full_n_segments.restype = c_int

# WHISPER_API int whisper_full_n_segments_from_state(struct whisper_state * state);
def whisper_full_n_segments_from_state(state: whisper_state_p) -> int:
    return _lib.whisper_full_n_segments_from_state(state)

_lib.whisper_full_n_segments_from_state.argtypes = [whisper_state_p]
_lib.whisper_full_n_segments_from_state.restype = c_int

# // Language id associated with the context's default state
# WHISPER_API int whisper_full_lang_id(struct whisper_context * ctx);
def whisper_full_lang_id(ctx: whisper_context_p) -> int:
    return _lib.whisper_full_lang_id(ctx)

_lib.whisper_full_lang_id.argtypes = [whisper_context_p]
_lib.whisper_full_lang_id.restype = c_int

# // Language id associated with the provided state
# WHISPER_API int whisper_full_lang_id_from_state(struct whisper_state * state);
def whisper_full_lang_id_from_state(state: whisper_state_p) -> int:
    return _lib.whisper_full_lang_id_from_state(state)

_lib.whisper_full_lang_id_from_state.argtypes = [whisper_state_p]
_lib.whisper_full_lang_id_from_state.restype = c_int

# // Get the start and end time of the specified segment
# WHISPER_API int64_t whisper_full_get_segment_t0           (struct whisper_context * ctx, int i_segment);
def whisper_full_get_segment_t0(ctx: whisper_context_p, i_segment: int) -> int:
    return _lib.whisper_full_get_segment_t0(ctx, i_segment)

_lib.whisper_full_get_segment_t0.argtypes = [whisper_context_p, c_int]
_lib.whisper_full_get_segment_t0.restype = c_int64

# WHISPER_API int64_t whisper_full_get_segment_t0_from_state(struct whisper_state * state, int i_segment);
def whisper_full_get_segment_t0_from_state(state: whisper_state_p, i_segment: int) -> int:
    return _lib.whisper_full_get_segment_t0_from_state(state, i_segment)

_lib.whisper_full_get_segment_t0_from_state.argtypes = [whisper_state_p, c_int]
_lib.whisper_full_get_segment_t0_from_state.restype = c_int64

# WHISPER_API int64_t whisper_full_get_segment_t1           (struct whisper_context * ctx, int i_segment);
def whisper_full_get_segment_t1(ctx: whisper_context_p, i_segment: int) -> int:
    return _lib.whisper_full_get_segment_t1(ctx, i_segment)

_lib.whisper_full_get_segment_t1.argtypes = [whisper_context_p, c_int]
_lib.whisper_full_get_segment_t1.restype = c_int64

# WHISPER_API int64_t whisper_full_get_segment_t1_from_state(struct whisper_state * state, int i_segment);
def whisper_full_get_segment_t1_from_state(state: whisper_state_p, i_segment: int) -> int:
    return _lib.whisper_full_get_segment_t1_from_state(state, i_segment)

_lib.whisper_full_get_segment_t1_from_state.argtypes = [whisper_state_p, c_int]
_lib.whisper_full_get_segment_t1_from_state.restype = c_int64

# // Get whether the next segment is predicted as a speaker turn
# WHISPER_API bool whisper_full_get_segment_speaker_turn_next(struct whisper_context * ctx, int i_segment);
def whisper_full_get_segment_speaker_turn_next(ctx: whisper_context_p, i_segment: int) -> bool:
    return _lib.whisper_full_get_segment_speaker_turn_next(ctx, i_segment)

_lib.whisper_full_get_segment_speaker_turn_next.argtypes = [whisper_context_p, c_int]
_lib.whisper_full_get_segment_speaker_turn_next.restype = c_bool

# // Get the text of the specified segment
# WHISPER_API const char * whisper_full_get_segment_text           (struct whisper_context * ctx, int i_segment);
def whisper_full_get_segment_text(ctx: whisper_context_p, i_segment: int) -> bytes:
    return _lib.whisper_full_get_segment_text(ctx, i_segment)

_lib.whisper_full_get_segment_text.argtypes = [whisper_context_p, c_int]
_lib.whisper_full_get_segment_text.restype = c_char_p

# WHISPER_API const char * whisper_full_get_segment_text_from_state(struct whisper_state * state, int i_segment);
def whisper_full_get_segment_text_from_state(state: whisper_state_p, i_segment: int) -> bytes:
    return _lib.whisper_full_get_segment_text_from_state(state, i_segment)

_lib.whisper_full_get_segment_text_from_state.argtypes = [whisper_state_p, c_int]
_lib.whisper_full_get_segment_text_from_state.restype = c_char_p

# // Get number of tokens in the specified segment
# WHISPER_API int whisper_full_n_tokens           (struct whisper_context * ctx, int i_segment);
def whisper_full_n_tokens(ctx: whisper_context_p, i_segment: int) -> int:
    return _lib.whisper_full_n_tokens(ctx, i_segment)

_lib.whisper_full_n_tokens.argtypes = [whisper_context_p, c_int]
_lib.whisper_full_n_tokens.restype = c_int

# WHISPER_API int whisper_full_n_tokens_from_state(struct whisper_state * state, int i_segment);
def whisper_full_n_tokens_from_state(state: whisper_state_p, i_segment: int) -> int:
    return _lib.whisper_full_n_tokens_from_state(state, i_segment)

_lib.whisper_full_n_tokens_from_state.argtypes = [whisper_state_p, c_int]
_lib.whisper_full_n_tokens_from_state.restype = c_int

# // Get the token text of the specified token in the specified segment
# WHISPER_API const char * whisper_full_get_token_text           (struct whisper_context * ctx, int i_segment, int i_token);
def whisper_full_get_token_text(ctx: whisper_context_p, i_segment: int, i_token: int) -> bytes:
    return _lib.whisper_full_get_token_text(ctx, i_segment, i_token)

_lib.whisper_full_get_token_text.argtypes = [whisper_context_p, c_int, c_int]
_lib.whisper_full_get_token_text.restype = c_char_p

# WHISPER_API const char * whisper_full_get_token_text_from_state(struct whisper_context * ctx, struct whisper_state * state, int i_segment, int i_token);
def whisper_full_get_token_text_from_state(
    ctx: whisper_context_p,
    state: whisper_state_p,
    i_segment: int,
    i_token: int
) -> bytes:
    return _lib.whisper_full_get_token_text_from_state(ctx, state, i_segment, i_token)

_lib.whisper_full_get_token_text_from_state.argtypes = [
    whisper_context_p,
    whisper_state_p,
    c_int,
    c_int
]
_lib.whisper_full_get_token_text_from_state.restype = c_char_p

# WHISPER_API whisper_token whisper_full_get_token_id           (struct whisper_context * ctx, int i_segment, int i_token);
def whisper_full_get_token_id(
    ctx: whisper_context_p,
    i_segment: int,
    i_token: int
) -> whisper_token:
    return _lib.whisper_full_get_token_id(ctx, i_segment, i_token)

_lib.whisper_full_get_token_id.argtypes = [whisper_context_p, c_int, c_int]
_lib.whisper_full_get_token_id.restype = whisper_token

# WHISPER_API whisper_token whisper_full_get_token_id_from_state(struct whisper_state * state, int i_segment, int i_token);
def whisper_full_get_token_id_from_state(
    state: whisper_state_p,
    i_segment: int,
    i_token: int
) -> whisper_token:
    return _lib.whisper_full_get_token_id_from_state(state, i_segment, i_token)

_lib.whisper_full_get_token_id_from_state.argtypes = [whisper_state_p, c_int, c_int]
_lib.whisper_full_get_token_id_from_state.restype = whisper_token

# // Get token data for the specified token in the specified segment
# // This contains probabilities, timestamps, etc.
# WHISPER_API whisper_token_data whisper_full_get_token_data           (struct whisper_context * ctx, int i_segment, int i_token);
def whisper_full_get_token_data(
    ctx: whisper_context_p,
    i_segment: int,
    i_token: int
) -> whisper_token_data:
    return _lib.whisper_full_get_token_data(ctx, i_segment, i_token)

_lib.whisper_full_get_token_data.argtypes = [whisper_context_p, c_int, c_int]
_lib.whisper_full_get_token_data.restype = whisper_token_data

# WHISPER_API whisper_token_data whisper_full_get_token_data_from_state(struct whisper_state * state, int i_segment, int i_token);
def whisper_full_get_token_data_from_state(
    state: whisper_state_p,
    i_segment: int,
    i_token: int
) -> whisper_token_data:
    return _lib.whisper_full_get_token_data_from_state(state, i_segment, i_token)

_lib.whisper_full_get_token_data_from_state.argtypes = [whisper_state_p, c_int, c_int]
_lib.whisper_full_get_token_data_from_state.restype = whisper_token_data

# // Get the probability of the specified token in the specified segment
# WHISPER_API float whisper_full_get_token_p           (struct whisper_context * ctx, int i_segment, int i_token);
def whisper_full_get_token_p(ctx: whisper_context_p, i_segment: int, i_token: int) -> float:
    return _lib.whisper_full_get_token_p(ctx, i_segment, i_token)

_lib.whisper_full_get_token_p.argtypes = [whisper_context_p, c_int, c_int]
_lib.whisper_full_get_token_p.restype = c_float

# WHISPER_API float whisper_full_get_token_p_from_state(struct whisper_state * state, int i_segment, int i_token);
def whisper_full_get_token_p_from_state(
    state: whisper_state_p,
    i_segment: int,
    i_token: int
) -> float:
    return _lib.whisper_full_get_token_p_from_state(state, i_segment, i_token)

_lib.whisper_full_get_token_p_from_state.argtypes = [whisper_state_p, c_int, c_int]
_lib.whisper_full_get_token_p_from_state.restype = c_float

# ////////////////////////////////////////////////////////////////////////////

# // Temporary helpers needed for exposing ggml interface

# WHISPER_API int          whisper_bench_memcpy          (int n_threads);
def whisper_bench_memcpy(n_threads: int) -> int:
    return _lib.whisper_bench_memcpy(n_threads)

_lib.whisper_bench_memcpy.argtypes = [c_int]
_lib.whisper_bench_memcpy.restype = c_int

# WHISPER_API const char * whisper_bench_memcpy_str      (int n_threads);
def whisper_bench_memcpy_str(n_threads: int) -> bytes:
    return _lib.whisper_bench_memcpy_str(n_threads)

_lib.whisper_bench_memcpy_str.argtypes = [c_int]
_lib.whisper_bench_memcpy_str.restype = c_char_p

# WHISPER_API int          whisper_bench_ggml_mul_mat    (int n_threads);
def whisper_bench_ggml_mul_mat(n_threads: int) -> int:
    return _lib.whisper_bench_ggml_mul_mat(n_threads)

_lib.whisper_bench_ggml_mul_mat.argtypes = [c_int]
_lib.whisper_bench_ggml_mul_mat.restype = c_int

# WHISPER_API const char * whisper_bench_ggml_mul_mat_str(int n_threads);
def whisper_bench_ggml_mul_mat_str(n_threads: int) -> bytes:
    return _lib.whisper_bench_ggml_mul_mat_str(n_threads)

# // Control logging output; default behavior is to print to stderr

# typedef void (*whisper_log_callback)(const char * line);
whisper_log_callback_fn_t=CFUNCTYPE(None, c_char_p)

# WHISPER_API void whisper_set_log_callback(whisper_log_callback callback);
def whisper_set_log_callback(callback: whisper_log_callback_fn_t) -> None:
    return _lib.whisper_set_log_callback(callback)

_lib.whisper_set_log_callback.argtypes = [whisper_log_callback_fn_t]
_lib.whisper_set_log_callback.restype = None
