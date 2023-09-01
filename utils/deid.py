import os
import cv2
import numpy as np

import torch
import torchaudio
import torchaudio.transforms as transforms

def GaussianBlur(image, kernel_size=(51, 51)):
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image

def Masking(image):
    h, w, _ = image.shape
    black_mask = np.zeros((h, w, 3), dtype=np.uint8)
    return black_mask

def DataReplace(image, source_image, target_mask):
    h, w, _ = image.shape
    mask = cv2.imread(target_mask, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    mask_resize = cv2.resize(mask, (w, h))
    dst = cv2.imread(source_image, cv2.IMREAD_COLOR).astype(np.uint8)
    dst_resize = cv2.resize(dst, (w, h))

    h_, w_ = dst_resize.shape[:2]
    crop = image[0:h_, 0:w_]
    cv2.copyTo(dst_resize, mask_resize, crop)    
    return crop

def PitchShift(audio_path, semitones):
    waveform, sample_rate = torchaudio.load(audio_path) # 입력 음성 파일 로드
    pitch_shift = transforms.PitchShift(sample_rate=sample_rate, n_steps=semitones) # 피치 변조 변환 함수 생성
    shifted_waveform = pitch_shift(waveform) # 음성 변조 적용

    torchaudio.save(audio_path, shifted_waveform, sample_rate) # 변조된 음성 파일 저장

def AddNoise(audio_path, noise_factor):
    waveform, sample_rate = torchaudio.load(audio_path)
    noise = torch.randn_like(waveform) * noise_factor
    noisy_audio = audio_path + noise
    
    torchaudio.save(audio_path, noisy_audio, sample_rate)

def Resample(audio_path, sampling_ratio):
    waveform, sample_rate = torchaudio.load(audio_path)  # 입력 음성 파일 로드
    
    # Resampling
    new_sample_rate = sampling_ratio * sample_rate
    resampler = transforms.Resample(sample_rate, new_sample_rate)
    resampled_waveform = resampler(waveform)
    
    # 새로운 샘플링 주파수로 변환된 음성을 저장
    torchaudio.save(audio_path, resampled_waveform, sample_rate)