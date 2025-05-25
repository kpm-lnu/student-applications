from reader.files_reader import read_image_file


def process_file(file_path: str, file_extension: str):
    ## TODO. Add other file types
    if file_extension == ".jpg":
        points, segments = read_image_file(file_path)
        return points, segments
