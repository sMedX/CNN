import os
import subprocess
from subprocess import call
import errno
import itertools
import sys
import random
import time

# environ
clas = os.environ['clas']
postproc = os.environ['postproc']
valid = os.environ['valid']
python = os.environ['python']

imagesDir = os.environ['images']

def main():
    preset = 'livertumors'
    deviceId = '0'##

    classCount = '2'
    r = '112'
    size = int(r) * 2
    label = preset + '.nrrd'
    mask = 'liver.nrrd'
    patient = 'patient.nrrd'

    samplesList = os.path.join(imagesDir, preset, 'samplesListRelative.txt') ##

    netDir = 'C:/Users/alkor\Downloads/20161014-091700-5cd6_epoch_0.2.tar/'
    deploy = os.path.join(netDir, 'deploy.prototxt')
    model = os.path.join(netDir, 'snapshot_iter_5926.caffemodel')
    start = [0, 0, 0]
    size = [512, 512, 1000]
    batchLength = '128'
    groupX = '3'
    groupY = '3'
    spacing = '0.8'

    sigma = '4.000000'
    sampleTrainList = 'Y:/NFS/nets/livertumors/a_grey_less_bg/train-cv-0-2.txt'
    sampleTestList =  'Y:/NFS/nets/livertumors/a_grey_less_bg/test-cv-0-2.txt'

    #apply
    with open(samplesList) as f:
        for line in f:
            line=line.replace('\n','')
            outputCNN='preset_'+netDir.split('/')[-1]+'.nrrd' 
            args=[clas, deploy, model, str(start[0]), str(start[1]), str(start[2]), str(size[0]), str(size[1]), str(size[2]), r,
            preset, spacing, batchLength, groupX, groupY, classCount, os.path.join(imagesDir, preset, line, patient),
            os.path.join(imagesDir, preset, line, mask), os.path.join(imagesDir, preset, line, outputCNN), deviceId]
            print args
            subprocess.call(args)

    validate(sampleTrainList, sampleTestList, label, outputCNN, '')
    
    #postprocess     
    with open(samplesList) as f:
        for line in f:
            line = line.replace('\n','')
            outputCNN='preset_'+netDir.split('/')[-1]+'_postproc.nrrd' 
            subprocess.call([postproc, '-image', os.path.join(imagesDir, preset, line, outputCNN), '-gaussianVariance', sigma, '-preset', preset])

    validate(sampleTrainList, sampleTestList, label, outputCNN, '')


def validate(samplesTrainListK, samplesTestListK, label, outputCNN, suffix):
    with open(samplesTrainListK) as f:
        for line in f:
            line = line.replace('\n','')
            subprocess.call([valid, '-testImage', os.path.join(imagesDir, preset, line, outputCNN), '-label', os.path.join(imagesDir, preset, line, label),
                             '-report', 'report-cnn-train-' + preset + '-' + suffix + '.csv'])
    with open(samplesTestListK) as f:
        for line in f:
            line = line.replace('\n','')
            subprocess.call([valid, '-testImage', os.path.join(imagesDir, preset, line, outputCNN), '-label', os.path.join(imagesDir, preset, line, label),
                             '-report', 'report-cnn-test-'  + preset + '-'+ suffix + '.csv'])
    
if __name__ == "__main__":
    main()
