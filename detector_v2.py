import argparse
import os
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import time
import tensorrt as trt

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

####################
## CLASSIFICATION ##
####################

def predict(model, img, batch_size):
    """ function for image classification"""
    img = cv2.resize(img, (128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = np.argmax(model.predict(x), axis=-1)
    #get percent of smoke prediction - perc_s
    perc_s = round(100*model.predict(x)[0][1], 1)
    #get percent of no smoke prediction - perc_n
    perc_n = round(100*model.predict(x)[0][0], 1)
    return preds[0], perc_s, perc_n

def classification(filename, image, model, batch_size):
    """ function for block image classification """
    h_im = image.shape[0]
    w_im = image.shape[1]
    #get name of file without extension
    name = os.path.splitext(os.path.basename(filename))[0]
    #create detection file
    detections = open('dets/' + name + '.txt', 'a')
    #size defines the square side of detection zone in image
    size = 416
    X_data = []

    while h+size <= h_im:
        while w+size <= w_im:
            img = image.copy()
            #crop needed piece of image
            img = img[h:h+size,w:w+size]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (416, 416))
            img = np.array(img)
            X_data.append(img)
            w = w+size

        h = h+size
        w = 0
    X_data = np.array(X_data)
    y_pred = model.predict(X_data, batch_size=batch_size)

    i = 0
    while h+size <= h_im:
        while w+size <= w_im:
            #get position of block relatively to image
            block_position = str('h: '+ h + ' h+size:'+h+size+', w: '+w+' w+size:'+w+size)
            pred_res = str('smoke %: ' + str(round(y_pred[i][0]*100, 2)) + 'no_smoke %' + str(round(y_pred[i][1]*100, 2)) )
            #write detection results
            detections.write(pred_res+'\n')
            line = "Block: " + block_position + pred_res
            _log('info', line) 

            w = w+size
            i+=1
        h = h+size
        w = 0

    detections.close()

#############################
## OBJECT DETECTION YOLOV4 ##
#############################

def get_engine(engine_path):
    #if a serialized engine exists, use it instead of building an engine.
    _log ('info', "Reading engine from file {}".format(engine_path))
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:

        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dims = engine.get_binding_shape(binding)
        
        # in case batch dimension is -1 (dynamic)
        if dims[0] < 0:
            size *= -1
        
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

# def plot_boxes_cv2(img, boxes, class_names=None, color=None):
#     import cv2
#     img = np.copy(img)
#     colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

#     def get_color(c, x, max_val):
#         ratio = float(x) / max_val * 5
#         i = int(math.floor(ratio))
#         j = int(math.ceil(ratio))
#         ratio = ratio - i
#         r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
#         return int(r * 255)

#     width = img.shape[1]
#     height = img.shape[0]
#     for i in range(len(boxes)):
#         box = boxes[i]
#         x1 = int(box[0] * width)
#         y1 = int(box[1] * height)
#         x2 = int(box[2] * width)
#         y2 = int(box[3] * height)

#         if color:
#             rgb = color
#         else:
#             rgb = (255, 0, 0)
#         if len(box) >= 7 and class_names:
#             cls_conf = box[5]
#             cls_id = box[6]
#             #print('%s: %f' % (class_names[cls_id], cls_conf))
#             classes = len(class_names)
#             offset = cls_id * 123457 % classes
#             red = get_color(2, offset, classes)
#             green = get_color(1, offset, classes)
#             blue = get_color(0, offset, classes)
#             if color is None:
#                 rgb = (red, green, blue)
#             img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
#         img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
#     return img

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def detect(context, buffers, image_src, image_size, num_classes, image_path):
    IN_IMAGE_H, IN_IMAGE_W = image_size

    #get name of file without extension
    name = os.path.splitext(os.path.basename(filename))[0]
    #create detection file
    detections = open('dets/' + name + '.txt', 'a')

    ta = time.time()
    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    img_in = np.ascontiguousarray(img_in)
    #_log('info', "Shape of the network input: ", img_in.shape)

    inputs, outputs, bindings, stream = buffers
    #_log('info', 'Length of inputs: ', len(inputs))
    inputs[0].host = img_in
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    #_logt('info', 'Len of outputs: ', len(trt_outputs))

    trt_outputs[0] = trt_outputs[0].reshape(1, -1, 1, 4)
    trt_outputs[1] = trt_outputs[1].reshape(1, -1, num_classes)

    tb = time.time()
    _log('info', 'TRT inference time: %f' % (tb - ta))
    boxes = post_processing(img_in, 0.4, 0.6, trt_outputs)

    # box[5] - confidence
    # box[6] index of object: 0 - smoke; 1 - fire
    line = 'Image: ' + filename + 'Object: ' + box[6] + 'Confidence: ' + box[5] 
    _log('info', line)
    detections.write(pred_res+'\n')
    detections.close()

    return boxes

def detect_trt_yolov4(engine_path, image_path, image_size):
    with get_engine(engine_path) as engine, engine.create_execution_context() as context:
        buffers = allocate_buffers(engine, 1)

        image_src = cv2.imread(image_path)
        IN_IMAGE_H, IN_IMAGE_W = image_size
        context.set_binding_shape(0, (1, 3, IN_IMAGE_H, IN_IMAGE_W))

        num_classes = 2
        namesfile = 'data/names'

        for i in range(2):  # This 'for' loop is for speed check
                            # Because the first iteration is usually longer
            boxes = detect(context, buffers, image_src, image_size, num_classes, image_path)

        class_names = load_class_names(namesfile)
        #plot_boxes_cv2(image_src, boxes[0], class_names=class_names)
#############################

def open_image(symlink):
    """ function for opening image from .npz format by symlink """
    #define path to image by symlink
    path = os.readlink(symlink)
    #image is in .npz format. open .npz format with numpy
    image = np.load(path)
    line = "Image was loaded by path: " + path
    _log('info', line)  
    return image


def check_symlinks(symlinksFolder, oldListDir):
    """ function for checking symlinks folder for new files """
    #get curent dirlist in folder
    dirList = os.listdir(symlinksFolder)
    #compare current dirlist with older one
    if len(dirList)>len(oldListDir):
        line = "Found new symlinks files. Count: " + str(len(dirList)-len(oldListDir))
        _log('info', line)
        #list listNewSymlinks contains symlinks that do not exist in oldListDir
        setDifference = set(dirList) - set(oldListDir)
        listNewSymlinks = list(setDifference)
    else:
        line = "No new symlinks files"
        _log('info', line)  
        listNewSymlinks = []
    return listNewSymlinks, dirList

def main():
    if len(sys.argv) == 5:
        # sys.argv[1] - detection/classification flag
        # sys.argv[2] - path to symlinks folder
        # sys.argv[3] - path to model
        # sys.argv[4] - batch size
        if not os.path.isdir(sys.argv[2]):
			_log('error', 'Symlinks folder cannot be found by path: %s' % sys.argv[2])
        
        symlinksFolder = sys.argv[2]
        MODEL_FILE = sys.argv[3]
        batch_size = sys.argv[4]

        # object detection or classification
        op_type = sys.argv[1]
        if op_type == "yolov4":
            line = "Chosen option: object detection by YOLOv4"
            _log('info', line)

        if op_type == "classification":
            line = "Chosen option: classification"
            _log('info', line)
    
    #check new current symlinks
    currentListDir = os.listdir(symlinksFolder)
    #change flag in file if previously it was set false
    file_flag = open('flag.txt', 'w')
    file_flag.write('true')
    file_flag.close()
    #before while set flag_ true
    flag_ = 'true'

    while (flag_ != 'false'):
        #condition - if in file flag was set false, it would finish current circle
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
                    classification(filename, image, model, batch_size)
            
            if op_type == "yolov4":
                for filename in listNewSymlinks:
                    line = "For image " + filename
                    _log('info', line)
                    detect_trt_yolov4(MODEL_FILE, os.readlink(filename), image_size)
        
        #check if flag was set false
        file_flag = open('flag.txt', 'r')
        flag_ = file_flag.readlines()[0]
        if flag_ == 'false':
            _log('info', 'detector was stopped by changing flag' )


if __name__ == '__main__':
	ec = main()
    sys.exit(ec)
