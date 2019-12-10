import pickle

def unpickle_to_dict(file):
        with open(file, 'rb') as fo:
            file_dict = pickle.load(fo, encoding='bytes')
        return file_dict