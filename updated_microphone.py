import pydub
import time
from queue import Queue
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from wav2vec2_inference import Wave2Vec2Inference
from indictrans2 import IndicTranslator

asr_trans_queue = Queue()
trans_tts_queue = Queue()

src_lang, tgt_lang = "hin_Deva", "ben_Beng"

executor = ThreadPoolExecutor(max_workers=60)  # Adjust max_workers as needed
lock = threading.Lock()  # Mutex for synchronized access to the queues

def tts(input_text, model):
    pass

def translate(input_text, model):
    global translator
    bn_translations = translator.single_translate(input_text, src_lang, tgt_lang)
    print("bengali_translation", bn_translations)
    with lock:
        trans_tts_queue.put(bn_translations)

def transcribe(wav_file, model):
    """
    wav_file: path of the.wav file
    """
    global asr
    print("transcription started.")
    transcription_start_time = time.time()
    result = asr.file_to_text(wav_file)
    print("Transcription: ", result)
    with lock:
        asr_trans_queue.put(result)
    print("Transcription time: ", time.time() - transcription_start_time)

def start():
    # Load the audio file
    audio = pydub.AudioSegment.from_mp3("/root/akarx/Solving_ASR/wav2vec_streaming/wav2vec2-live/modi_legit.wav")

    # Set frame duration (in milliseconds)
    frame_duration_ms = 3000  # Change this according to your requirement

    # Calculate total number of frames
    total_frames = len(audio) // frame_duration_ms

    # Simulate real-time frame processing
    for i in range(total_frames):
        # Extract the current frame
        start_time = i * frame_duration_ms
        end_time = (i + 1) * frame_duration_ms
        sound_chunk = audio[start_time:end_time]
        import os
        file_num = len(os.listdir("audio_outputs")) + 1
        file_name = f"audio_outputs/output_{file_num}.wav"
        sound_chunk = sound_chunk.set_frame_rate(16000)
        sound_chunk.export(file_name, format="wav")
        print("Audio chunk processed and passed to a thread")
        future = executor.submit(transcribe, f"./{file_name}", asr)
        future.result()  # Wait for the transcribe task to complete
        time.sleep(frame_duration_ms / 1000)  # Simulate real-time processing
        future = executor.submit(translate, asr_trans_queue.get(), translator)
        future.result()  # Wait for the translation task to complete

asr = Wave2Vec2Inference("ai4bharat/indicwav2vec-hindi", use_lm_if_possible=True)
translator = IndicTranslator("indic-indic", "ai4bharat/indictrans2-indic-indic-1B")

start()
