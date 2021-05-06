import argparse
import os
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import time

currentListDir = os.listdir(symlinksFolder)

def _log(severity,tag):
    logfile = "fileLog.txt"
    loglevels = ['error','warning','info','debug']
    level_conf = 0
    level_in = 0

    try:
        level_in = loglevels.index(severity)
    except ValueError:
        level_in = 2
    if level_conf >= level_in: 
        line = "%s [%s] (%s): %s" % (time.strftime('%d-%m-%Y %H:%M:%S',time.localtime()),str(os.getpid()),severity,tag)
        log = open(logfile,'a')
        log.write(line)
        log.write("\r\n")
        log.close()

def predict(model, img):
    """ function for image classification"""
    img = cv2.resize(img, (128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = np.argmax(model.predict(x), axis=-1)
    perc_s = round(100*model.predict(x)[0][1], 1)
    perc_n = round(100*model.predict(x)[0][0], 1)
    return preds[0], perc_s, perc_n

def classification(image, model):
    """ function for block image classification """
    h_im = image.shape[0]
    w_im = image.shape[1]
    size = 416
    while h+size <= h_im:
        while w+size <= w_im:
            img = image.copy()
            img = img[h:h+size,w:w+size]
            preds, perc_s, perc_n = predict(model, img)
            block_position = str('h: '+ h + ' h+size:'+h+size+', w: '+w+' w+size:'+w+size)
            pred_res = str('smoke %: ' + perc_s + 'no_smoke %' + perc_n)
            line = "Block: " + block_position + pred_res
            _log('info', line) 
            i = i+1
            w = w+size
        h = h+size
        w = 0


def open_image(symlink):
    """ function for opening image from .npz format by symlink """
    path = os.readlink(symlink)
    image = np.load(path)
    line = "Image was loaded by path: " + path
    _log('info', line)  
    return image

def check_symlinks(symlinksFolder, oldListDir):
    """ function for checking symlinks folder for new files """
    dirList = os.listdir(symlinksFolder)
    if len(dirList)>len(oldListDir):
        line = "Found new symlinks files. Count: " + str(len(dirList)-len(oldListDir))
        _log('info', line)   
        setDifference = set(dirList) - set(oldListDir)
        listNewSymlinks = list(setDifference)
    else:
        line = "No new symlinks files"
        _log('info', line)   
        listNewSymlinks = []
    return listNewSymlinks, dirList

def main():
    if len(sys.argv) == 4:
        # sys.argv[1] - detection/classification flag
        # sys.argv[2] - path to symlinks folder
        # sys.argv[3] - path to model
        if not os.path.isdir(sys.argv[2]):
			_log('error', 'Symlinks folder cannot be found by path: %s' % sys.argv[2])
        
        symlinksFolder = sys.argv[2]
        MODEL_FILE = sys.argv[3]

        # object detection or classification
        op_type = sys.argv[1]
        if op_type == "yolov4":
            line = "Chosen option: object detection by YOLOv4"
            _log('info', line) 
        if op_type == "classification":
            line = "Chosen option: classification"
            _log('info', line)
            model = load_model(MODEL_FILE, compile = True)
            _log('info', 'Model file was loaded')
    
    flag_ = 'true'
    while (flag_ != 'false'):
        listNewSymlinks, listDir = check_symlinks(symlinksFolder, currentListDir)
        currentListDir = listDir

        if len(listNewSymlinks)==0:
            _log('warning', 'No new symlinks: continue') 
            continue()
        else:
            if op_type == "classification":
                for filename in listNewSymlinks:
                    line = "For image " + filename
                    _log('info', line)
                    image = open_image(filename, fileLog)
                    classification(image, model)
        
        file_flag = open('flag.txt', 'r')
        flag_ = file_flag.readlines()[0]
        if f flag_ == 'false':
            _log('info', 'detector was stopped by changing flag')


if __name__ == '__main__':
	ec = main()
    sys.exit(ec)
