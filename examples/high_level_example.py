from whisper_cpp import Whisper

whisper = Whisper("/../models/ggml-large.bin")

whisper.transcribe("samples.wav", diarize=True)

whisper.output(output_csv=True, output_jsn=True, output_lrc=True, output_srt=True, output_txt=True, output_vtt=True, log_score=True)
