import os
from os import walk

def folder(path):    
    os.mkdir(path)
    return path

def gettingfilenames(path):    
    f = []
    for (dirpath, dirnames, filenames) in walk(path):
        f.extend(filenames)
    return f