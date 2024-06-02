import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor
import librosa  # Import librosa
import torch 
from datasets import load_dataset
import io
import time 
import librosa
import os
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

hpu = torch.device('hpu')
cpu = torch.device('cpu')

class Wave2Vec2Inference:
    def __init__(self, model_name, hotwords=[], use_lm_if_possible=True, use_gpu=True):
        self.device = "hpu" if use_gpu else "cpu"
        if use_lm_if_possible:            
            self.processor = AutoProcessor.from_pretrained(model_name)
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(model_name)
        self.model.to(self.device)
        #self.model = ht.hpu.wrap_in_hpu_graph(self.model)
        self.hotwords = hotwords
        self.use_lm_if_possible = use_lm_if_possible

    def buffer_to_text(self, audio_buffer):
        if len(audio_buffer) == 0:
            return ""

        inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="pt", padding=True)
       
        start = time.time()
        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device),
                                attention_mask=inputs.attention_mask.to(self.device)).logits            
        print("total time taken", time.time() - start)
        if hasattr(self.processor, 'decoder') and self.use_lm_if_possible:
            transcription = \
                self.processor.decode(logits[0].cpu().numpy(),                                      
                                      hotwords=self.hotwords,
                                      output_word_offsets=True,                                      
                                   )                             
            confidence = transcription.lm_score / len(transcription.text.split(" "))
            transcription = transcription.text       
        else:
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            confidence = self.confidence_score(logits, predicted_ids)

        return transcription
        #return transcription, confidence   

    def confidence_score(self, logits, predicted_ids):
        scores = torch.nn.functional.softmax(logits, dim=-1)                                                           
        pred_scores = scores.gather(-1, predicted_ids.unsqueeze(-1))[:, :, 0]
        mask = torch.logical_and(
            predicted_ids.not_equal(self.processor.tokenizer.word_delimiter_token_id), 
            predicted_ids.not_equal(self.processor.tokenizer.pad_token_id))

        character_scores = pred_scores.masked_select(mask)
        total_average = torch.sum(character_scores) / len(character_scores)
        return total_average

    def file_to_text(self, filename):
        # Load audio with librosa at 16kHz
        audio_input, _ = librosa.load(filename, sr=16000)
        return self.buffer_to_text(audio_input)
    
if __name__ == "__main__":
    print("Model test")
    asr = Wave2Vec2Inference("ai4bharat/indicwav2vec-hindi", use_lm_if_possible=True)
    text = asr.file_to_text("/root/akarx/Bhashini_Pipeline/TTS/Indic-TTS/inference/modi_legit.wav")
    print(text)
    text = asr.file_to_text("/root/akarx/Habana_Gaudi_Bhashini/samples/common_voice_hi_23796066.wav")
    print(text)
