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

http://librosa.github.io/librosa/generated/librosa.feature.chroma_cqt.html
'''

# Feature extraction example
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt




filename = r'1.wav'
# Load the example clip
y, sr = librosa.load(filename)
y = y-0.97*y



from scipy import signal
import numpy as np
import math
# b, a = signal.butter(4, 16000/sr*0.5, "highpass") # lowpass
# 阶数；最大纹波允许低于通频带中的单位增益。以分贝表示，以正数表示；频率(Hz)/奈奎斯特频率（采样率*0.5）
b, a = signal.cheby1(4, 5, 16000/sr*0.5, "highpass") # lowpass
yt  = signal.filtfilt(b, a, y)
y = yt

S,phase = librosa.magphase(librosa.stft(y=y))
contrast = librosa.feature.spectral_contrast(y=y,sr=sr)

print('rms.shape:',contrast.shape)

plt.figure()
plt.subplot(211)
librosa.display.specshow(contrast, x_axis='time')
plt.colorbar()
plt.ylabel('Frequency bands')
plt.title('Spectral contrast')
plt.subplot(212)
librosa.display.specshow(librosa.amplitude_to_db(S,ref=np.max),y_axis='log',x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Power spectrogram')
plt.tight_layout()
plt.show()



from scipy import signal
import numpy as np
import math
# b, a = signal.butter(4, 16000/sr*0.5, "highpass") # lowpass
# 阶数；最大纹波允许低于通频带中的单位增益。以分贝表示，以正数表示；频率(Hz)/奈奎斯特频率（采样率*0.5）
b, a = signal.cheby1(4, 5, 16000/sr*0.5, "highpass") # lowpass
yt  = signal.filtfilt(b, a, y)
plt.subplot(121)
plt.plot(y)
plt.subplot(122)
x = np.linspace(0,10,sr)
print(type(x),yt.shape[0])
plt.plot(yt)
plt.title("lp")

plt.show()
