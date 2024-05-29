import torch 
from datasets import load_dataset
import io
import time 
import librosa
import os
#from TTS.utils.synthesizer import Synthesizer
from indictrans2 import IndicTranslator
#from Indic_TTS.inference.src.inference import TextToSpeechEngine
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from scipy.io.wavfile import write
import soundfile as sf 
DEFAULT_SAMPLING_RATE = 16000

os.environ['PT_HPU_ENABLE_GENERIC_STREAM'] = '1'
os.environ['PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES'] = '0'

import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import time
import os
import librosa 
import numpy as np
import pydub
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
import torchaudio
import soundfile as sf
import torchaudio.functional as F  # For resampling the audio 
from habana_frameworks.torch.hpex.experimental.transformer_engine import recipe
import habana_frameworks.torch.hpex.experimental.transformer_engine as te
import habana_quantization_toolkit
import soundfile as sf
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs
#from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
import soundfile as sf

#adapt_transformers_to_gaudi()
is_long_audio = True
hpu = torch.device('hpu')
cpu = torch.device('cpu')

# Load the audio file
audio_path = "/root/akarx/Bhashini_Pipeline/TTS/Indic-TTS/inference/modi_legit.wav"
speech, rate = librosa.load(audio_path, sr=16000)

# Initialize the processor and model

print("=================================================")
print("\n=================================================")

model_id = "ai4bharat/indicwav2vec-hindi"
processor = Wav2Vec2Processor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)
model = model.eval().to(torch.device('hpu'))


# Define the threshold in seconds
# For descent result 1000000 for hindi

if (is_long_audio):
    threshold_ms = 1000000 # Example threshold of 1000 milliseconds
    threshold_s = threshold_ms / 10.0

    # Calculate the duration of each segment
    segment_duration_s = threshold_s
    num_segments = int(np.ceil(len(speech) / segment_duration_s))
    # print(num_segments)
    # Divide the audio into segments
    num_segments = 4
    segments = np.array_split(speech, num_segments)

    # Initialize an empty list to store the transcriptions
    all_transcriptions = []

    import time

    start = time.time()
    for i, segment in enumerate(segments):
        # Process each segment
        start_time = time.time()
        
        # Convert the segment to the required format
        input_values = processor(segment, sampling_rate=16000, return_tensors='pt', padding="max_length", max_length=200000).input_values
        
        # Move the input values to the HPU device
        input_values = input_values.to(torch.device('hpu'), non_blocking=True)
        
        # Perform inference
        with torch.no_grad():
            logits = model(input_values).logits.to(torch.device('cpu'), non_blocking=True)

        time.sleep(0.05)
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        
        # Append the transcription to the list
        all_transcriptions.append(transcription)
        all_transcriptions.append(" ")
        
        #print(f"Segment {i+1} processed in {time.time() - start_time:.2f} seconds.")
        #print(f"Transcription: {transcription}\n")

    end = time.time()
    # Combine the transcriptions from each segment
    final_transcription = " ".join(all_transcriptions)

    print("=================================================")
    print("\n=================================================\n")

    print("WARMING UP ")
    print("\nFinal Transcription:", final_transcription)
    print("\ntotal time taken", end - start)

    print("=================================================")
    print("\n=================================================\n")

    start = time.time()

    for i, segment in enumerate(segments):
        # Process each segment
        start_time = time.time()
        
        # Convert the segment to the required format
        input_values = processor(segment, sampling_rate=16000, return_tensors='pt', padding="max_length", max_length=200000).input_values
        
        # Move the input values to the HPU device
        input_values = input_values.to(torch.device('hpu'), non_blocking=True)
        
        # Perform inference
        with torch.no_grad():
            logits = model(input_values).logits.to(torch.device('cpu'), non_blocking=True)

        time.sleep(0.05)
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        
        # Append the transcription to the list
        all_transcriptions.append(transcription)
        
        #print(f"Segment {i+1} processed in {time.time() - start_time:.2f} seconds.")
        #print(f"Transcription: {transcription}\n")

    end = time.time()

    print("=================================================")
    print("\n=================================================\n")

    # Combine the transcriptions from each segment
    final_transcription = " ".join(all_transcriptions)
    print("ACTUAL OUTPUT")
    print("\nFinal Transcription:", final_transcription)
    print("\ntotal time taken", end - start)

else:
    input_values = processor(speech, sampling_rate = 16000, return_tensors = 'pt',  padding="max_length", max_length=200000).input_values
    input_values = input_values.to(hpu, non_blocking=True)

    start = time.time()
    with torch.autocast(device_type="hpu"):
        logits = model(input_values).logits.to(cpu, non_blocking=True)
    end = time.time()

    print("first total time taken", end-start)
    predicted_ids = torch.argmax(logits, dim =-1)

    transcriptions = processor.decode(predicted_ids[0])

    print(transcriptions)

    import time
    start = time.time()

    input_values = processor(speech, sampling_rate = 16000, return_tensors = 'pt',  padding="max_length", max_length=200000).input_values
    input_values = input_values.to(hpu, non_blocking=True)

    with torch.autocast(device_type="hpu"):
        logits = model(input_values).logits.to(cpu, non_blocking=True)
    end = time.time()

    print("second total time taken", end-start)
    time.sleep(0.05)
    predicted_ids = torch.argmax(logits, dim =-1)

    transcriptions = processor.decode(predicted_ids[0])

    print(transcriptions)

if(not is_long_audio):
    final_transcription = transcriptions

print("=================================================")
print("\n=================================================\n")

translator = IndicTranslator("indic-indic", "ai4bharat/indictrans2-indic-indic-1B")

src_lang, tgt_lang = "hin_Deva", "ben_Beng"
#ben_translations = translator.batch_translate(sentences, src_lang, tgt_lang)

print("TRANSLATION STARTED FROM HINDI TO BENGALI")
start_time = time.time()
ben_single_translations = translator.single_translate(final_transcription, src_lang, tgt_lang)
end_time = time.time()
print("=================================================")
print("\n=================================================\n")

print("Modi ji's voice in bengali:", ben_single_translations)
print("Total time taken for translation", end_time - start_time)
print("=================================================")
print("\n=================================================\n")

#complete_text = "\n".join(ben_translations)


import io

from TTS.utils.synthesizer import Synthesizer
from src.inference import TextToSpeechEngine
import scipy.io.wavfile

# Initialize Bengali model

DEFAULT_SAMPLING_RATE = 16000

lang = "bn"
ben_model  = Synthesizer(
    tts_checkpoint=f'checkpoints/{lang}/fastpitch/best_model.pth',
    tts_config_path=f'checkpoints/{lang}/fastpitch/config.json',
    tts_speakers_file=f'checkpoints/{lang}/fastpitch/speakers.pth',
    # tts_speakers_file=None,
    tts_languages_file=None,
    vocoder_checkpoint=f'checkpoints/{lang}/hifigan/best_model.pth',
    vocoder_config=f'checkpoints/{lang}/hifigan/config.json',
    encoder_checkpoint="",
    encoder_config="",
    use_cuda=False,
)

# Setup TTS Engine

models = {
    "bn": ben_model
}
engine = TextToSpeechEngine(models)

# Bengali TTS inference

print("=================================================")
print("\n=================================================\n")

print("CONVERTING TRANSLATED TEXT INTO SPEECH \n\n")

start_time = time.time()
bengali_raw_audio = engine.infer_from_text(
    input_text=ben_single_translations,
    lang="bn",
    speaker_name="male"
)
end_time = time.time()

print("Total time taken", end_time - start_time)
print("=================================================")
print("\n=================================================\n")

byte_io = io.BytesIO()
scipy.io.wavfile.write(byte_io, DEFAULT_SAMPLING_RATE, bengali_raw_audio)

print("SAVING AUDIO FILE :)")

with open("bengali_audio.wav", "wb") as f:
    f.write(byte_io.read())
