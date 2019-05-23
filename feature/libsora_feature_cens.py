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

http://librosa.github.io/librosa/generated/librosa.feature.chroma_cens.html#
'''

# Feature extraction example
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
filename = r'1.wav'
# Load the example clip
y, sr = librosa.load(filename)

chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)


plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
librosa.display.specshow(chroma_cq, y_axis='chroma')
plt.title('chroma_cq')
plt.colorbar()
plt.subplot(2, 1, 2)
librosa.display.specshow(chroma_cens, y_axis='chroma', x_axis='time')
plt.title(' chroma_cens')
plt.colorbar()
plt.tight_layout()
plt.show()