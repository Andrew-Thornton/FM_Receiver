# FM Receiver Project

## Description

This is a fun fm receiver project where I aim to use an rfsoc to receive an FM signal of my favourite radio station (Triple J 105.5MHz), downsample it, de-modulate it, split the left and right channels, and then play it over a speaker of some kind.

## Simulation of the project
Currently there is a simulation of the project. What I have done so far is that you can download your favourite song which you own the license to in Wav format. Then save it in 
py_sim/input_files/my_wav.wav
Then you can run 
```
python wav_shortener.py
```
To drop it down to one second of audio.

To then upsample your audio to 245.76MHz use 
```
python generate_fm_signal.py
```

Then to simulate demodulating the signal there are 2 scripts
```
python cic_decimate.py
python demodulate.py
```

This will then output your audio in py_sim/output_files/mono_out.wav and py_sim/output_files/stereo_out.wav

## Running the project

Currently the project is a work in progress but to pull in all sub repos please run from the top of the repo
```
git submodule update --init --recursive
``` 