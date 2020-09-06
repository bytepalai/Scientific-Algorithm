import subprocess

subprocess.call(['../darknet/darknet', 'detect', '../darknet/cfg/yolov3.cfg', '../darknet/yolov3.weights', './new.jpg'])
