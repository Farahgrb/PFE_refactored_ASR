import librosa
import torchaudio.transforms as T
import torch
import torchaudio
from speechbrain.pretrained import WhisperASR
from dotenv import dotenv_values
import arabic_reshaper


env_vars = dotenv_values()



path = "whisper"


asr_model = WhisperASR.from_hparams(source=path, savedir=path)

def process_audio_chunk(chunk, device):
    tensor_chunk = chunk.to(device)
    if chunk.size(1) != 16000:
        resampled = T.Resample(chunk.size(1), 16000, dtype=torch.float)(tensor_chunk)
    else:
        resampled = tensor_chunk
    return resampled

def stereo_to_mono(audio_data):
    # Mix down the stereo audio to mono
    mono_audio = librosa.to_mono(audio_data.T)
    return mono_audio


def convert_to_mono_and_resample(audio_data, sample_rate):
    # Convert stereo to mono if it has two channels
    if audio_data.ndim > 1:
        audio_data = stereo_to_mono(audio_data)
    # Resample the audio to 16,000 Hz
    target_sample_rate = 16000
    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)
    return audio_data

def segment_audio(audio_data, segment_duration):

    audio, sample_rate = librosa.load(audio_data, sr=None)
    audio = convert_to_mono_and_resample(audio, sample_rate)
    segment_samples = int(segment_duration * 16000)
    segments = []
    for i in range(0, len(audio), segment_samples):

        segment = audio[i:i+segment_samples]
        torchaudio.save('resampled.wav', torch.from_numpy(segment).unsqueeze(0), 16000)
        chunk='resampled.wav'
        list_transc = asr_model.transcribe_file(chunk)
        transcription = list_transc[0]
        segments.append(transcription)

    return segments

def transcribe(wav):
    
    transcriptions = []
    
    # Split the audio into chunks
    transcriptions = segment_audio(wav.file, env_vars["CHUNK_DURATION"])

    complete_transcription = " ".join(transcriptions)

    reshaped_text = arabic_reshaper.reshape(complete_transcription)
    print(reshaped_text)

    return { 'Transcription': complete_transcription }

