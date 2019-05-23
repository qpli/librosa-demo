#! python
# encoding: utf-8
'''
@author: LiQingpei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 707901160@qq.com
@software: pyCharm
@file: libsora1.py
@time: 2019/5/23 18:07
@desc:
'''

# Beat tracking example
from __future__ import print_function
import librosa

# 1. Get the file path to the included audio example 'wav_lqpA/20181210092119.wav'
filename = r'1.wav'

# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
y, sr = librosa.load(filename,sr=None)

# 3. Run the default beat tracker
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

# 4. Convert the frame indices of beat events into timestamps
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

print('Saving output to beat_times.csv')
librosa.output.times_csv('beat_times.csv', beat_times)



