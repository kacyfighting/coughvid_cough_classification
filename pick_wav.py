import os
from scipy.io import wavfile
import json

data_folder = 'D:\\Zhang\\public_dataset_wav\\public_dataset_wav_json'
data_folder1 = 'D:\\Zhang\\public_dataset_wav\\exe'
filter = [".json"]
filter_wav = ".wav"

num_max = 200
num = 0
for maindir, subdir, file_name_list in os.walk(data_folder):
    for filename in file_name_list:
        apath = os.path.join(maindir, filename)
        portion = os.path.splitext(filename)
        ext = portion[1]
        if ext in filter:
            with open(data_folder + '\\' + filename, 'r') as load_f:
                load_dict = json.load(load_f)
                # if float(load_dict['cough_detected']) > 0.95:
                #     num = num + 1
                #     if num <= num_max:
                #         print(portion[0]+filter_wav)

                if float(load_dict['cough_detected']) < 0.10:
                    num = num + 1
                    if num <= num_max:
                        print(portion[0]+filter_wav)

