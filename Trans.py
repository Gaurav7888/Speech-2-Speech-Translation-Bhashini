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

activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU]

translator = IndicTranslator("indic-indic", "ai4bharat/indictrans2-indic-indic-1B")

src_lang, tgt_lang = "hin_Deva", "ben_Beng"
#ben_translations = translator.batch_translate(sentences, src_lang, tgt_lang)
final_transcription = "इसमें समाज के सभी वर्गों का प्रतिनिधित्व होता है एक प्रकार से   उनके लिए सहकारिता सबके साथ से सबके कल्याण का सही मार्ग   थीसिर्फ महाराष्ट्र ी नहीं अटल जी की सरकार में मंत्री रहते हुए   उन्होंने देश के अनेक क्षेत्रों में सहकारिता को बढ़ावा दिया उसके लिए प्रयास किया   इसमें समाज के सभी वर्गों का प्रतिनिधित्व होता है एक प्रकार से उनके लिए सहकारिता सबके साथ से सबके कल्याण का सही मार्ग थीसिर्फ महाराष्ट्र ी नहीं अटल जी की सरकार में मंत्री रहते हुए उन्होंने देश के अनेक क्षेत्रों में सहकारिता को बढ़ावा दिया उसके लिए प्रयास किया"

print("TRANSLATION STARTED FROM HINDI TO BENGALI")
with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=10, active=5, repeat=1),
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('logs_new')) as profiler:
    for i in range(30):
        ben_single_translations = translator.single_translate(final_transcription, src_lang, tgt_lang)
        profiler.step()
        htcore.mark_step()


print("=================================================")
print("\n=================================================\n")

print("Modi ji's voice in bengali:", ben_single_translations)
print("=================================================")
print("\n=================================================\n")
