from torch.utils.data import Dataset
from feature_extraction import get_mfccs_and_phones, get_mfccs_and_spectrogram
import glob

# MFCC and phone needed for train1
class Net1TimitData(Dataset):
    phone_vocab_size = 61
    def __init__(self, path):
        self.wav_file_list = glob.glob(path)
        self.size = len(self.wav_file_list)

        self.phone_vocab = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
                            'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
                            'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
                            'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
                            'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
    # Return mfccs & phones
    def __getitem__(self, index):
        wav_file_name = self.wav_file_list[index]
        mfccs, phns = get_mfccs_and_phones(wav_file=wav_file_name, phone_vocab=self.phone_vocab)
        item = {'mfccs': mfccs, 'phns': phns}
        return item

    def __len__(self):
        return self.size

# MFCC and PPG (no transcription required)
# PPG comes from Net1
class Net2Data(Dataset):
    # def __init__(self, path):
    #     # path is path of list
    #     self.wav_file_list = []
    #     with open(path, 'r') as f:
    #         for line in f:
    #             self.wav_file_list.append(line)
    #
    #     self.size = len(self.wav_file_list)

    def __init__(self, list):
        self.wav_file_list = list
        self.size = len(self.wav_file_list)

    def __getitem__(self, index):
        wav_file_name = self.wav_file_list[index]
        mfccs, y_spec, y_mel = get_mfccs_and_spectrogram(wav_file_name, trim=True, random_crop=False)
        item = {'mfccs': mfccs, 'y_spec': y_spec, 'y_mel': y_mel}
        return item

    def __len__(self):
        return self.size

class Net3Data(Dataset):
    def __init__(self, list):
        self.wav_file_list = list
        self.size = len(self.wav_file_list)

    def __getitem__(self, index):
        wav_file_name = self.wav_file_list[index]
        mfccs, y_spec, y_mel = get_mfccs_and_spectrogram(wav_file_name, trim=True, random_crop=False)
        item = {'mfccs': mfccs, 'y_spec': y_spec, 'y_mel': y_mel}
        return item

    def __len__(self):
        return self.size

# Korean speaker, train1: MFCC and phone needed for train1
class Net1AvmData(Dataset):
    def __init__(self, path):
        self.wav_file_list = glob.glob(path)
        self.phone_vocab = ["g",  "n",  "d",  "l",  "m",  "b",  "s",  "-", "j",   "q",
                "k", "t", "p", "h", "x", "w", "f", "c", "z", "A",
                "o", "O", "U", "u", "E", "a", "e", "1", "2", "3",
                "4", "5", "6", "7", "8", "9", "[", "]", "<", ">",
                "G", "N", "D", "L", "M", "B", "0", "K", ";;",";", "sp", "*",
                "$", "?", "!","#"]
        self.size = len(self.wav_file_list)

    def __getitem__(self, index):
        wav_file_name = self.wav_file_list[index]
        mfccs, phns = get_mfccs_and_phones(wav_file=wav_file_name, phone_vocab=self.phone_vocab)
        item = {'mfccs': mfccs, 'phns': phns}
        return item

    def __len__(self):
        return self.size

