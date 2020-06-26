import logging
import sys
from datetime import datetime
import os
import stat
import codecs
from importlib import reload
from subprocess import call
os.chmod('Onetime_Installations.sh', 777)
os.system('pwd')
os.system('ls-al')
os.system('cat/etc/os-release')
from wand.image import Image
import random
import cv2
import numpy as np
from segmentation import ourapproach
from keras.models import load_model
from sklearn.metrics import accuracy_score
from src.creatingfolder import folder, gettingfilenames
from os import walk
from TOCclassifier import prediction, loading_model1
from Tessaract import tocparsing
import pandas as pd
import gc
from jsoncreation import main
import boto3
import botocore
import botocore.session
import shutil
import argparse

logging.basicConfig(filename='segmentation_log.log', level=logging.DEBUG)
bucket_name = 'sm-poc-mya-csa-ctrct'
path = '/ContractPDF/'

def pdftoimg(inputfile):    
    model = load_model('DoTnet.h5')
    print('Load model completed')
    logging.info("Directory creation starttime:::%s" % datetime.now())
    if os.path.exists(inputfile[:-4]):
        None
    else:
        os.makedirs(inputfile[:-4])
    logging.info("Directory creation endtime:::%s" % datetime.now()
    logging.info('directory creation completed')
    modeltoc = loading_model1()
 
    with(Image(filename=inputfile, resolution=300)) as source:
        print('input file name====' + inputfile)
        images = source.sequence
        pages = 4
        logging.info(pages)

        FileText = []
        TOC = []
        for i in range(0, pages):
            try:
                logging.info("page creation starttime:::%s" % datetime.now())
                Image(images[i]).save(filename=str(inputfile[:-4] + '/' + str(i) + '.png'))
                label = prediction("%s/%s.png" % (inputfile[:-4], i), modeltoc, i)
                 
                if i < 13:

                    if label == 1:
                        Image(images[i]).save(filename=str('expresw2/TOC' + '/' + str(i) + '.png'))
                        Toc = tocparsing("%s/%s.png" % (inputfile[:-4], i))
                        print(Toc)
                        TOC.extend(Toc)
                        tocdf = pd.DataFrame(Toc)
                        tocdf.to_csv('%s%stoc.csv' % (i, inputfile[:-4]))

                    else:
                        pageentity = ourapproach(model, 8, str(inputfile[:-4] + '/' + str(i) + '.png'), i, inputfile[:-4])
                        FileText.extend(pageentity)
                else:
                    pageentity = ourapproach(model, 8, str(inputfile[:-4] + '/' + str(i) + '.png'), i, inputfile[:-4])
                    
                    FileText.extend(pageentity)
                
                logging.info("page creation endtime:::%s" % datetime.now())
                 
            except Exception as e:
                print(e)
                 
        Tocdf = pd.DataFrame(TOC)
        Tocdf.to_csv('TOC%s.csv' % inputfile[:-4])
        with open("%stoc.txt" % inputfile[:-4], "w") as output1:
            for line1 in TOC:
                output1.write(str(line1))
        with open("%s.txt" % inputfile[:-4], "w") as output:
            for line in FileText:
                output.write(str(line))
        
    return TOC

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='')
    parser.add_argument('--bucketname', type=str, default='')
   
    args=parser.parse_args()
    s3 = boto3.resource('s3')
    
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.all():
        key = obj.key
                
    try:
        for obj in bucket.objects.all():
            key = obj.key
            
            s3.Bucket(bucket_name).download_file(key, key)
            print('Downloaded file name======' + key)
            logging.info("pdf to img" + key + "start time::%s" % datetime.now())
            pdftoimg(key)
            logging.info("pdf to img" + key + "end time::%s" % datetime.now())
            s4 = boto3.client('s3')

            s4.upload_file(key[:-4] + '.txt', bucket_name,key[:-4] + '.txt')
           
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist." + e)
        else:
            raise