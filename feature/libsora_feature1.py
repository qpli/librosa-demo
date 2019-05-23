#! python
# encoding: utf-8
'''
@author: LiQingpei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 707901160@qq.com
@software: pyCharm
@file: libsora_feature1.py
@time: 2019/5/23 18:18
@desc:
'''

# Feature extraction example
import numpy as np
import librosa
import librosa.display
filename = r'1.wav'
# Load the example clip
y, sr = librosa.load(filename)

# Set the hop length; at 22050 Hz, 512 samples ~= 23ms
hop_length = 512

# Separate harmonics and percussives into two waveforms
y_harmonic, y_percussive = librosa.effects.hpss(y)

# Beat track on the percussive signal
tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
                                             sr=sr)

# Compute MFCC features from the raw signal
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

# And the first-order differences (delta features)
mfcc_delta = librosa.feature.delta(mfcc)

# Stack and synchronize between beat events
# This time, we'll use the mean value (default) instead of median
beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),
                                    beat_frames)

# Compute chroma features from the harmonic signal
chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                        sr=sr)

# Aggregate chroma features between beat events
# We'll use the median value of each feature between beat frames
beat_chroma = librosa.util.sync(chromagram,
                                beat_frames,
                                aggregate=np.median)

# Finally, stack all beat-synchronous features together
beat_features = np.vstack([beat_chroma, beat_mfcc_delta])


mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1000)



import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()




m_slaney = librosa.feature.mfcc(y=y, sr=sr, dct_type=2)
m_htk = librosa.feature.mfcc(y=y, sr=sr, dct_type=3)
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
librosa.display.specshow(m_slaney, x_axis='time')
plt.title('RASTAMAT / Auditory toolbox (dct_type=2)')
plt.colorbar()
plt.subplot(2, 1, 2)
librosa.display.specshow(m_htk, x_axis='time')
plt.title('HTK-style (dct_type=3)')
plt.colorbar()
plt.tight_layout()
plt.show()

S = np.abs(librosa.stft(y))
chroma1 = librosa.feature.chroma_stft(S=S, sr=sr)

S = np.abs(librosa.stft(y, n_fft=4096))**2
chroma2 = librosa.feature.chroma_stft(S=S, sr=sr)
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
librosa.display.specshow(chroma1, x_axis='time')
plt.title('energy (magnitude) spectrum instead of power spectrogram')
plt.colorbar()
plt.subplot(2, 1, 2)
librosa.display.specshow(chroma2, x_axis='time')
plt.title(' pre-computed power spectrogram with a larger frame')
plt.colorbar()
plt.tight_layout()
plt.show()