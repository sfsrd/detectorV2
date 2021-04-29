import argparse
import os
import cv2
import numpy as np

symlinksFolder = 'path-to/symlinks'
currentListDir = os.listdir(symlinksFolder)
fileLog = open("fileLog.txt", "w") 

""" load model for classification"""
MODEL_FILE = 'path-to/model_ResNet152.model'
model = load_model(MODEL_FILE, compile = True)


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

def classification(image, model, fileLog):
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
            fileLog.write("Block: " + block_position + pred_res + "\n") 
            i = i+1
            w = w+size
        h = h+size
        w = 0


def open_image(symlink, fileLog):
    """ function for opening image from .npz format by symlink """
    path = os.readlink(symlink)
    image = np.load(path)
    fileLog.write("Image was loaded by path: " + path + "\n") 
    return image

def check_symlinks(symlinksFolder, oldListDir, fileLog):
    """ function for checking symlinks folder for new files """
    dirList = os.listdir(symlinksFolder)
    if len(dirList)>len(oldListDir):
        fileLog.write("Found new symlinks files. Count: " + len(dirList)>len(oldListDir)+ "\n") 
        setDifference = set(dirList) - set(oldListDir)
        listNewSymlinks = list(setDifference)
    else:
        fileLog.write("No new symlinks files \n") 
        listNewSymlinks = []
    return listNewSymlinks, dirList


parser = argparse.ArgumentParser()
parser.add_argument("--option", choices=["yolov4", "classification"], required=True, type=str)
args = parser.parse_args()
op_type = args.option
if op_type == "yolov4":
    fileLog.write("Chosen option: object detection by YOLOv4 \n") 
if op_type == "classification":
    fileLog.write("Chosen option: classification \n") 

listNewSymlinks, listDir = check_symlinks(symlinksFolder, currentListDir, fileLog)
currentListDir = listDir

if len(listNewSymlinks)==0:
    fileLog.write("No new symlinks: break \n")
    break()
else:
    if op_type == "yolov4":
        for filename in listNewSymlinks:
            fileLog.write("For image " + filename + "\n")
            image = open_image(filename, fileLog)

fileLog.close() 
