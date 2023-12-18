from whisper_cpp import Whisper

# whisper = Whisper("/Volumes/ExternalData/AI/audio/whisper.cpp/models/ggml-large.bin", verbose=False)
whisper = Whisper("/path/to/whisper.cpp/models/ggml-large.bin")

# whisper.transcribe("examples/samples/spanish.wav", translate=False, language="es")
# whisper.transcribe("examples/samples/spanish.wav", translate=True, language="es")
whisper.transcribe("examples/samples/english.wav", translate=False, language="en")

whisper.output(output_csv=True, output_jsn=True, output_lrc=True, output_srt=True, output_txt=True, output_vtt=True, log_score=True)
