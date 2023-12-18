import os
import multiprocessing
import json
from ctypes import (
    c_float,
    c_void_p,
    cast,
    pointer,
)

from . import whisper_cpp
from .utils import (
    suppress_stdout_stderr,
    read_wav,
    to_timestamp,
    to_lrc_timestamp,
    escape_double_quotes_and_backslashes,
    estimate_diarization_speaker,
)
from .whisper_callback import (
    whisper_print_segment_callback,
    whisper_print_progress_callback,
    whisper_print_encoder_begin_callback,
)

class Whisper:
    """High-level Python wrapper for a whisper.cpp model."""
    def __init__(
        self,
        model_path: str,
        openvino_encode_device: str = "CPU",
        verbose: bool = True,
        print_colors: bool = False,
        no_timestamps: bool = False,
    ) -> None:
        """Load a whisper.cpp model from `model_path`

        Args:
            model_path (str):                 model path
            openvino_encode_device (str):     the OpenVINO device used for encode inference.
            verbose (bool, optional):         enable debug mode (eg. dump process info and log_mel).
            print_colors (bool, optional):    print colors.
            no_timestamps (bool, optional):   do not print timestamps.
        """
        self.model_path = model_path
        self.openvino_encode_device = openvino_encode_device
        self.verbose = verbose
        self.print_colors = print_colors
        self.no_timestamps = no_timestamps
        self.success = False
        # for output
        self.file_name = None
        self.diarize = False
        self.tinydiarize = False
        self.pcmf32 = []
        self.pcmf32s = []
        self.language = "en"
        self.translate = False

        if self.model_path is None:
            raise ValueError("model_path must be specified!")

        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {self.model_path}")

        if self.verbose:
            self.debug_mode = True
        else:
            self.debug_mode = False

        if self.verbose:
            self.whisper_ctx = whisper_cpp.whisper_init_from_file(self.model_path.encode("utf-8"))
        else:
            with suppress_stdout_stderr():
                self.whisper_ctx = whisper_cpp.whisper_init_from_file(self.model_path.encode("utf-8"))

        if self.whisper_ctx is None:
            raise RuntimeError("Failed to load model!")

        whisper_cpp.whisper_ctx_init_openvino_encoder(
            self.whisper_ctx,
            self.model_path.encode("utf-8"),
            self.openvino_encode_device.encode("utf-8"),
            None
        )

    def transcribe(
        self,
        fname_inp: str,
        n_threads: int = min(4, multiprocessing.cpu_count()),
        n_processors: int = 1,
        offset_t_ms: int = 0,
        duration_ms: int = 0,
        max_context: int = -1,
        max_len: int = 0,
        best_of: int = 2,
        beam_size: int = -1,
        word_thold: float = 0.01,
        entropy_thold: float = 2.40,
        logprob_thold: float = -1.0,
        language: str = "en",
        detect_language: bool = False,
        prompt: str = "",
        translate: bool = False,
        diarize: bool = False,
        tinydiarize: bool = False,
        split_on_word: bool = False,
        no_fallback: bool = False,
        print_special: bool = False,
        print_progress: bool = True,
    ) -> None:
        """Transcribe audio from `fname_inp` to text

        Args:
            fname_inp (str):                  input WAV file path

            n_threads(int):                   number of threads to use during computation
            n_processors (int):               number of processors to use during computation
            offset_t_ms (int):                time offset in milliseconds
            duration_ms (int):                duration of audio to process in milliseconds
            max_context (int):                maximum number of text context tokens to store
            max_len (int):                    maximum segment length in characters
            best_of (int):                    number of best candidates to keep
            beam_size (int):                  beam size for beam search
            word_thold (float):               word timestamp probability threshold
            entropy_thold (float):            entropy threshold for decoder fail
            logprob_thold (float):            log probability threshold for decoder fail
            language (str):                   spoken language ('auto' for auto-detect).
            detect_language (bool, optional): exit after automatically detecting language.
            prompt (str):                     initial prompt
            translate (bool, optional):       translate from source language to english.
            diarize (bool, optional):         stereo audio diarization.
            tinydiarize (bool, optional):     enable tinydiarize (requires a tdrz model).
            split_on_word (bool, optional):   split on word rather than on token.
            no_fallback (bool, optional):     do not use temperature fallback while decoding.
            print_special (bool, optional):   print special tokens.
            print_progress (bool, optional):  print progress.
        """
        if fname_inp is None:
            raise ValueError("fname_inp must be specified!")

        if not os.path.exists(fname_inp):
            raise ValueError(f"Input file path does not exist: {fname_inp}")

        if not read_wav(fname_inp, self.pcmf32, self.pcmf32s, diarize):
            raise RuntimeError("Failed to read WAV file!")

        if language != "auto" and whisper_cpp.whisper_lang_id(language.encode("utf-8")) == -1:
            raise ValueError(f"Language not supported: {language}")

        if diarize and tinydiarize:
            raise ValueError("Cannot enable both diarize and tinydiarize!")

        # print system information
        if self.verbose:
            system_flags = whisper_cpp.whisper_print_system_info().decode("utf-8")
            print(f"\nsystem information: total cpu core count = {multiprocessing.cpu_count()}; threads count in use = {n_threads*n_processors}")
            print(f"system flags: {system_flags}")

        if not whisper_cpp.whisper_is_multilingual(self.whisper_ctx):
            if language != "en" or translate:
                language = "en"
                translate = False
                if self.verbose:
                    print(f"{__name__}: WARNING: multilingual model disabled, so disabling translation and setting language to english")

        if detect_language:
            language = "auto"
            if self.verbose:
                print(f"{__name__}: WARNING: language detection enabled, setting language to auto")

        if self.verbose:
            task = "translate" if translate else "transcribe"
            print(f"{__name__}: processing {fname_inp} ({len(self.pcmf32)} samples, {len(self.pcmf32)/whisper_cpp.WHISPER_SAMPLE_RATE} seconds), {n_threads} threads, {n_processors} processors, lang = {language}, task = {task} timestamps = {not self.no_timestamps}")

        self.file_name, _ = os.path.splitext(fname_inp)
        self.diarize = diarize
        self.tinydiarize = tinydiarize
        self.language = language
        self.translate = translate

        wparams = whisper_cpp.whisper_full_default_params(whisper_cpp.WHISPER_SAMPLING_GREEDY)

        wparams.strategy = whisper_cpp.WHISPER_SAMPLING_BEAM_SEARCH if beam_size > 1 else whisper_cpp.WHISPER_SAMPLING_GREEDY

        wparams.n_threads = n_threads
        wparams.n_max_text_ctx = max_context if max_context >= 0 else wparams.n_max_text_ctx
        wparams.offset_ms = offset_t_ms
        wparams.duration_ms = duration_ms

        wparams.translate = translate

        wparams.print_special = print_special
        wparams.print_progress = print_progress
        wparams.print_timestamps = not self.no_timestamps

        wparams.token_timestamps = max_len > 0
        wparams.thold_pt = word_thold
        wparams.max_len = max_len
        wparams.split_on_word = split_on_word

        wparams.debug_mode = self.debug_mode

        wparams.tdrz_enable = tinydiarize

        wparams.initial_prompt = prompt.encode("utf-8") if len(prompt) > 0 else None

        wparams.language = language.encode("utf-8")
        wparams.detect_language = detect_language

        wparams.temperature_inc = 0.0 if no_fallback else 0.4
        wparams.entropy_thold = entropy_thold
        wparams.logprob_thold = logprob_thold

        wparams.greedy.best_of = best_of
        wparams.beam_search.beam_size = beam_size

        pcmf32_array = (c_float * len(self.pcmf32))(*(i for i in self.pcmf32))

        if self.verbose:
            # prepare process callback
            progress_callback_user_data = whisper_cpp.whisper_progress_callback_user_data(
                progress_prev=0,
                progress_step=5,
            )

            wparams.progress_callback = whisper_cpp.whisper_progress_callback_fn_t(whisper_print_progress_callback)
            wparams.progress_callback_user_data = cast(pointer(progress_callback_user_data), c_void_p)

            # prepare encoder begin callback
            encoder_begin_callback_user_data = whisper_cpp.whisper_encoder_begin_callback_user_data(
                is_aborted=False,
            )

            wparams.encoder_begin_callback = whisper_cpp.whisper_encoder_begin_callback_fn_t(whisper_print_encoder_begin_callback)
            wparams.encoder_begin_callback_user_data = cast(pointer(encoder_begin_callback_user_data), c_void_p)

            # prepare segment callback
            segment_callback_user_data = whisper_cpp.whisper_segment_callback_user_data(
                no_timestamps=self.no_timestamps,
                print_colors=self.print_colors,
                print_special=print_special,
                tinydiarize=tinydiarize,
                tdrz_speaker_turn=" [SPEAKER_TURN]".encode("utf-8")
            )
            wparams.new_segment_callback = whisper_cpp.whisper_new_segment_callback_fn_t(whisper_print_segment_callback)
            wparams.new_segment_callback_user_data = cast(pointer(segment_callback_user_data), c_void_p)

        if whisper_cpp.whisper_full_parallel(
            ctx=self.whisper_ctx,
            params=wparams,
            samples=pcmf32_array,
            n_samples=len(self.pcmf32),
            n_processors=n_processors) != 0:
            raise RuntimeError("Failed to process audio!")

        self.success = True

        if self.verbose:
            whisper_cpp.whisper_print_timings(self.whisper_ctx)

    def output(
        self,
        fname_out: str = None,
        offset_n: int = 0,
        output_txt: bool = True,
        output_vtt: bool = False,
        output_srt: bool = False,
        output_lrc: bool = False,
        output_csv: bool = False,
        output_jsn: bool = False,
        log_score: bool = False,
    ) -> str:
        """Output the results
        Args:
            fname_out (str):                  output file path (without file extension)

            offset_n (int):                   segment index offset
            output_txt (bool, optional):      output result in a text file.
            output_vtt (bool, optional):      output result in a vtt file.
            output_srt (bool, optional):      output result in a srt file.
            output_lrc (bool, optional):      output result in a lrc file.
            output_csv (bool, optional):      output result in a csv file.
            output_jsn (bool, optional):      output result in a jsn file.
            log_score (bool, optional):       log best decoder scores of tokens.
        """
        if not self.success:
            print("Transcribe failed or not called yet!")
            return None

        if fname_out is None:
            fname_out = self.file_name

        if output_txt:
            output_txt_file = f"{fname_out}.txt"
            self.__output_txt(output_txt_file)

        if output_vtt:
            output_vtt_file = f"{fname_out}.vtt"
            self.__output_vtt(output_vtt_file)

        if output_srt:
            output_srt_file = f"{fname_out}.srt"
            self.__output_srt(output_srt_file, offset_n)

        if output_lrc:
            output_lrc_file = f"{fname_out}.lrc"
            self.__output_lrc(output_lrc_file)

        if output_csv:
            output_csv_file = f"{fname_out}.csv"
            self.__output_csv(output_csv_file)

        if output_jsn:
            output_jsn_file = f"{fname_out}.json"
            self.__output_jsn(output_jsn_file)

        if log_score:
            output_score_file = f"{fname_out}.score.txt"
            self.__output_score(output_score_file)

        return self.__output_str()

    def __output_txt(self, file_name: str) -> None:
        print(f"saving output to '{file_name}'")

        with open(file_name, "w", encoding="utf-8") as f:
            n_segments = whisper_cpp.whisper_full_n_segments(self.whisper_ctx)

            for i in range(n_segments):
                text = whisper_cpp.whisper_full_get_segment_text(self.whisper_ctx, i)
                speaker = ""

                if self.diarize and len(self.pcmf32s[0]) > 0:
                    time_start = whisper_cpp.whisper_full_get_segment_t0(self.whisper_ctx, i)
                    time_stop = whisper_cpp.whisper_full_get_segment_t1(self.whisper_ctx, i)
                    speaker = estimate_diarization_speaker(self.pcmf32s, time_start, time_stop, False)

                f.write(f"{speaker} {text.decode('utf-8')}\n")

    def __output_vtt(self, file_name: str) -> None:
        print(f"saving output to '{file_name}'")

        with open(file_name, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")

            n_segments = whisper_cpp.whisper_full_n_segments(self.whisper_ctx)

            for i in range(n_segments):
                text = whisper_cpp.whisper_full_get_segment_text(self.whisper_ctx, i)
                time_start = whisper_cpp.whisper_full_get_segment_t0(self.whisper_ctx, i)
                time_stop = whisper_cpp.whisper_full_get_segment_t1(self.whisper_ctx, i)
                speaker = ""

                if self.diarize and len(self.pcmf32s[0]) > 0:
                    speaker = estimate_diarization_speaker(self.pcmf32s, time_start, time_stop, True)
                    speaker = f"<v Speaker {speaker}>"

                f.write(f"{to_timestamp(time_start)} --> {to_timestamp(time_stop)}\n")
                f.write(f"{speaker} {text.decode('utf-8')}\n\n")

    def __output_srt(self, file_name: str, offset_n: int) -> None:
        print(f"saving output to '{file_name}'")

        with open(file_name, "w", encoding="utf-8") as f:
            n_segments = whisper_cpp.whisper_full_n_segments(self.whisper_ctx)

            for i in range(n_segments):
                text = whisper_cpp.whisper_full_get_segment_text(self.whisper_ctx, i)
                time_start = whisper_cpp.whisper_full_get_segment_t0(self.whisper_ctx, i)
                time_stop = whisper_cpp.whisper_full_get_segment_t1(self.whisper_ctx, i)
                speaker = ""

                if self.diarize and len(self.pcmf32s[0]) > 0:
                    speaker = estimate_diarization_speaker(self.pcmf32s, time_start, time_stop, False)

                f.write(f"{i + 1 + offset_n}\n")
                f.write(f"{to_timestamp(time_start, True)} --> {to_timestamp(time_stop, True)}\n")
                f.write(f"{speaker} {text.decode('utf-8')}\n\n")

    def __output_lrc(self, file_name: str) -> None:
        print(f"saving output to '{file_name}'")

        with open(file_name, "w", encoding="utf-8") as f:
            f.write("[by:whisper_cpp_python]\n")

            n_segments = whisper_cpp.whisper_full_n_segments(self.whisper_ctx)

            for i in range(n_segments):
                text = whisper_cpp.whisper_full_get_segment_text(self.whisper_ctx, i)
                time_start = whisper_cpp.whisper_full_get_segment_t0(self.whisper_ctx, i)
                speaker = ""

                if self.diarize and len(self.pcmf32s[0]) > 0:
                    time_stop = whisper_cpp.whisper_full_get_segment_t1(self.whisper_ctx, i)
                    speaker = estimate_diarization_speaker(self.pcmf32s, time_start, time_stop, False)

                f.write(f"[{to_lrc_timestamp(time_start)}] {speaker} {text.decode('utf-8')}\n")

    def __output_csv(self, file_name: str) -> None:
        print(f"saving output to '{file_name}'")

        with open(file_name, "w", encoding="utf-8") as f:
            n_segments = whisper_cpp.whisper_full_n_segments(self.whisper_ctx)
            f.write("start,end,")
            if self.diarize and len(self.pcmf32s[0]) > 0:
                f.write("speaker,")
            f.write("text\n")

            for i in range(n_segments):
                text = whisper_cpp.whisper_full_get_segment_text(self.whisper_ctx, i)
                time_start = whisper_cpp.whisper_full_get_segment_t0(self.whisper_ctx, i)
                time_stop = whisper_cpp.whisper_full_get_segment_t1(self.whisper_ctx, i)
                text_escape = escape_double_quotes_and_backslashes(text.decode('utf-8'))

                f.write(f"{time_start * 10},{time_stop * 10},")

                if self.diarize and len(self.pcmf32s[0]) > 0:
                    speaker = estimate_diarization_speaker(self.pcmf32s, time_start, time_stop, True)
                    f.write(f"{speaker},")

                f.write(f"\"{text_escape}\"\n")

    def __output_jsn(self, file_name: str) -> None:
        file_dict = {
            "systeminfo": whisper_cpp.whisper_print_system_info().decode("utf-8"),
            "model": {
                "type": whisper_cpp.whisper_model_type_readable(self.whisper_ctx).decode("utf-8"),
                "multilingual": whisper_cpp.whisper_is_multilingual(self.whisper_ctx),
                "vocab": whisper_cpp.whisper_model_n_vocab(self.whisper_ctx),
                "audio": {
                    "ctx": whisper_cpp.whisper_model_n_audio_ctx(self.whisper_ctx),
                    "state": whisper_cpp.whisper_model_n_audio_state(self.whisper_ctx),
                    "head": whisper_cpp.whisper_model_n_audio_head(self.whisper_ctx),
                    "layer": whisper_cpp.whisper_model_n_audio_layer(self.whisper_ctx),
                },
                "text": {
                    "ctx": whisper_cpp.whisper_model_n_text_ctx(self.whisper_ctx),
                    "state": whisper_cpp.whisper_model_n_text_state(self.whisper_ctx),
                    "head": whisper_cpp.whisper_model_n_text_head(self.whisper_ctx),
                    "layer": whisper_cpp.whisper_model_n_text_layer(self.whisper_ctx),
                },
                "mels": whisper_cpp.whisper_model_n_mels(self.whisper_ctx),
                "ftype": whisper_cpp.whisper_model_ftype(self.whisper_ctx),
            },
            "params": {
                "model": self.model_path,
                "language": self.language,
                "translate": self.translate,
            },
            "result": {
                "language": whisper_cpp.whisper_lang_str(whisper_cpp.whisper_full_lang_id(self.whisper_ctx)).decode("utf-8"),
            },
            "transcription": [],
        }

        n_segments = whisper_cpp.whisper_full_n_segments(self.whisper_ctx)

        for i in range(n_segments):
            text = whisper_cpp.whisper_full_get_segment_text(self.whisper_ctx, i)
            time_start = whisper_cpp.whisper_full_get_segment_t0(self.whisper_ctx, i)
            time_stop = whisper_cpp.whisper_full_get_segment_t1(self.whisper_ctx, i)

            segment_dict = {
                "timestamps": {
                    "from": to_timestamp(time_start, True),
                    "to": to_timestamp(time_stop, True),
                },
                "offset": {
                    "from": time_start * 10,
                    "to": time_stop * 10,
                },
                "text": text.decode('utf-8'),
            }

            if self.diarize and len(self.pcmf32s[0]) > 0:
                speaker = estimate_diarization_speaker(self.pcmf32s, time_start, time_stop, True)
                segment_dict.update({"speaker": speaker})

            if self.tinydiarize:
                segment_dict.update({"speaker_turn_next": whisper_cpp.whisper_full_get_segment_speaker_turn_next(self.whisper_ctx, i)})

            file_dict["transcription"].append(segment_dict)

        # write to json file
        print(f"saving output to '{file_name}'")

        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(file_dict, f, indent=4)

    def __output_score(self, file_name: str) -> None:
        print(f"saving output to '{file_name}'")

        with open(file_name, "w", encoding="utf-8") as f:
            n_segments = whisper_cpp.whisper_full_n_segments(self.whisper_ctx)

            for i in range(n_segments):
                n_tokens = whisper_cpp.whisper_full_n_tokens(self.whisper_ctx, i)
                for j in range(n_tokens):
                    token = whisper_cpp.whisper_full_get_token_text(self.whisper_ctx, i, j)
                    probability = whisper_cpp.whisper_full_get_token_p(self.whisper_ctx, i, j)
                    f.write(f"{token.decode('utf-8')}\t{probability}\n")

    def __output_str(self) -> str:
        result = ""
        n_segments = whisper_cpp.whisper_full_n_segments(self.whisper_ctx)
        for i in range(n_segments):
            text = whisper_cpp.whisper_full_get_segment_text(self.whisper_ctx, i)
            result = result + text.decode('utf-8')

        return result

    def __del__(self):
        if hasattr(self, "whisper_ctx") and self.whisper_ctx is not None:
            whisper_cpp.whisper_free(self.whisper_ctx)
            self.whisper_ctx = None
