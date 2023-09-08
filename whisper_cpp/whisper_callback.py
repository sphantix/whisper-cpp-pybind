from ctypes import (
    c_void_p,
    POINTER,
    cast,
)

from .whisper_cpp import (
    whisper_context_p,
    whisper_state_p,
    whisper_segment_callback_user_data,
    whisper_progress_callback_user_data,
    whisper_encoder_begin_callback_user_data,
    whisper_full_n_segments,
    whisper_full_get_segment_t0,
    whisper_full_get_segment_t1,
    whisper_full_n_tokens,
    whisper_full_get_token_id,
    whisper_token_eot,
    whisper_full_get_token_text,
    whisper_full_get_token_p,
    whisper_full_get_segment_text,
    whisper_full_get_segment_speaker_turn_next,
)

from .utils import (
    to_timestamp,
    get_color,
)

def whisper_print_segment_callback(
    whiper_context: whisper_context_p,
    whisper_state: whisper_state_p,
    n_new: int,
    user_data: c_void_p
) -> None:
    data = cast(user_data, POINTER(whisper_segment_callback_user_data))

    n_segments = whisper_full_n_segments(whiper_context)

    s0 = n_segments - n_new

    if s0 == 0:
        print("\n")

    for i in range(s0, n_segments):
        if not data.contents.no_timestamps:
            t0 = whisper_full_get_segment_t0(whiper_context, i)
            t1 = whisper_full_get_segment_t1(whiper_context, i)

            print(f"[{to_timestamp(t0)} --> {to_timestamp(t1)}]  ")

            if data.contents.print_colors:
                for j in range(0, whisper_full_n_tokens(whiper_context, i)):
                    if not data.contents.print_special:
                        if whisper_full_get_token_id(whiper_context, i, j) >= whisper_token_eot(whiper_context):
                            continue

                    text = whisper_full_get_token_text(whiper_context, i, j)
                    p = whisper_full_get_token_p(whiper_context, i, j)

                    color = get_color(p)
                    print(f"{color}{text}\033[0m")
            else:
                print(whisper_full_get_segment_text(whiper_context, i))

        if data.contents.tinydiarize:
            if whisper_full_get_segment_speaker_turn_next(whiper_context, i):
                print(f"{data.contents.tdrz_speaker_turn}")

        if not data.contents.no_timestamps:
            print("\n")

def whisper_print_progress_callback(
    whiper_context: whisper_context_p,
    whisper_state: whisper_state_p,
    progress: int,
    user_data: c_void_p
) -> None:
    data = cast(user_data, POINTER(whisper_progress_callback_user_data))
    if progress >= data.contents.progress_prev + data.contents.progress_step:
        data.contents.progress_prev = progress
        print(f"Progress: {progress}%")

def whisper_print_encoder_begin_callback(
    whiper_context: whisper_context_p,
    whisper_state: whisper_state_p,
    user_data: c_void_p
) -> bool:
    data = cast(user_data, POINTER(whisper_encoder_begin_callback_user_data))
    return not data.contents.is_aborted
