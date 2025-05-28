import os
import subprocess

def convert_mp4_to_wav(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.wav")
            command = f"ffmpeg -i \"{input_path}\" -q:a 0 -map a \"{output_path}\""
            subprocess.run(command, shell=True)
            print(f"Converted {filename} to WAV")

input_dev_directory = "./MELD.Raw/dev_splits_complete/"
output_dev_directory = "./dev_set2"

convert_mp4_to_wav(input_dev_directory, output_dev_directory)
