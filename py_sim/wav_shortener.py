import wave

input_file = "./input_files/my_wav.wav"
output_file = "./input_files/my_wav_short.wav"

start_time = 22.0  # seconds
end_time = 23.0    # seconds

with wave.open(input_file, "rb") as wav_in:
    params = wav_in.getparams()
    framerate = wav_in.getframerate()

    start_frame = int(start_time * framerate)
    end_frame = int(end_time * framerate)

    wav_in.setpos(start_frame)
    frames_to_read = end_frame - start_frame
    audio_data = wav_in.readframes(frames_to_read)

with wave.open(output_file, "wb") as wav_out:
    wav_out.setparams(params)
    wav_out.writeframes(audio_data)

print(f"Saved {output_file} from {start_time}s to {end_time}s")