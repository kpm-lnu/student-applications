from pydub import AudioSegment


def audio_cut(input_file, start_time, end_time, output_file):

    audio = AudioSegment.from_file(input_file)

    cut_audio = audio[start_time:end_time]

    cut_audio.export(output_file, format="wav")


input_path = "videoplayback.wav"
start = 5000 
end = 10000 
output_path = "poshliad.wav"

audio_cut(input_path, start, end, output_path)

