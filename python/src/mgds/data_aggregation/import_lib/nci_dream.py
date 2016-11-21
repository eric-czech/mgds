
import os

CACHE_DIR = '/Users/eczech/.synapseCache'


def get_all_dream_files():
    res = []
    for root, dirs, files in os.walk(CACHE_DIR):
        for file in files:
            if file.endswith(".txt"):
                res.append(os.path.join(root, file))
    return res


def get_file(search_string):
    for file in get_all_dream_files():
        if search_string in file:
            return file
    raise ValueError('Search string "{}" not found in local NIC DREAM file names'.format(search_string))


