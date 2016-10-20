# cuts image by tiles
import sys
import SimpleITK as sitk
import numpy as np
import os
import os.path
import subprocess
import scipy.misc
from PIL import Image

#creates 2.5d images as rgb images
def cutImageByTiles2p5d(imageSitk, labelSitk, maskSitk, r, stride, strideNegative, n, tilesFolder,  comment = ''):
    #zyx ordered
    image = sitk.GetArrayFromImage(imageSitk) #todo explicit uint8 type
    label = sitk.GetArrayFromImage(labelSitk)
    mask = sitk.GetArrayFromImage(maskSitk)

    sz = np.array(image.shape)

    indices = np.array(mask.nonzero())[::, ::stride]

    #find bounding box
    l = np.amin(indices, axis=1)
    u = np.amax(indices, axis=1)

    #crop around bounding+r
    cropL = l-r
    cropL[cropL<0]=0

    cropU = u+r
    cropU[cropU >= sz] = sz[cropU >= sz] - 1

    image = image[cropL[0]:cropU[0], cropL[1]:cropU[1], cropL[2]:cropU[2]]
    label = label[cropL[0]:cropU[0], cropL[1]:cropU[1], cropL[2]:cropU[2]]
    mask = mask[cropL[0]:cropU[0], cropL[1]:cropU[1], cropL[2]:cropU[2]]
    print 'new shape ', image.shape

    #pad
    # wtf with +1?
    padL = r-l+1
    padL[padL < 0] = 0

    padU = u+r-sz+1
    padU[padU < 0] = 0

    pad = ((padL[0],padU[0]),(padL[1],padU[1]),(padL[2],padU[2]))

    image = np.pad(image, pad , mode='constant')
    label = np.pad(label, pad, mode='constant')
    mask = np.pad(mask, pad, mode='constant')

    print 'new shape ', image.shape
    indices = np.array(mask.nonzero())[:, ::stride]
    indices = list(indices.T)

    print 'index count ', len(indices)

    for iLabel in [0,1]:
        try:
            os.makedirs(os.path.join(tilesFolder, n, str(iLabel)))
        except Exception as e:
            print e

    negCount = 0
    for index in indices:
        print 'index ', index, '\n'
        iLabel = label[index[0], index[1], index[2]]
        if iLabel == 0:
            negCount += 1
            if negCount % strideNegative != 0:
                continue

        print index
        tile1 = image[index[0], index[1]-r:index[1]+r, index[2]-r:index[2]+r]
        tile2 = image[index[0]-r:index[0]+r, index[1], index[2]-r:index[2]+r]
        tile3 = image[index[0]-r:index[0]+r, index[1]-r:index[1]+r, index[2]]

        print index[1]-r, index[1]+r
        print index[2] - r, index[2]+r
        print tile1.shape

        rgbArray = np.zeros((2*r,2*r,3), 'uint8')
        rgbArray[..., 0] = tile1
        rgbArray[..., 1] = tile2
        rgbArray[..., 2] = tile3
        img = Image.fromarray(rgbArray)
        path = os.path.join(tilesFolder, n, str(iLabel), '_'+str(index[0])+'_'+str(index[1])+'_'+str(index[2])+'_'
                            +comment+'.png')
        print path
        img.save(path)


if __name__ == '__main__':
    preprocessExe = os.environ['preproc']

    #print str(sys.argv)

    imageName, labelName, maskName, outputFolder, listFile = '', '', '', '', ''
    radius, stride, preset, strideNegative, spacingXY = None,None,'', None, 0.0
    try:
        imageName = sys.argv[sys.argv.index('--imageName')+1]
        labelName = sys.argv[sys.argv.index('--labelName1')+1]
        maskName = sys.argv[sys.argv.index('--maskName')+1]
        outputFolder = sys.argv[sys.argv.index('--outputFolder')+1]
        listFile = sys.argv[sys.argv.index('--listFile')+1]

        radius = int(sys.argv[sys.argv.index('--radius')+1])
        stride = int(sys.argv[sys.argv.index('--stride')+1].split(' ')[0]) #todo
        preset = sys.argv[sys.argv.index('--preset')+1]
        strideNegative = int(sys.argv[sys.argv.index('--strideNegative')+1])
        spacingXY = float(sys.argv[sys.argv.index('--spacingXY')+1])
    except Exception as e:
        print e
        exit(1)

    f = open(listFile, 'r')
    inputDirs = [i.replace('\n','') for i in f.readlines()]
    f.close()

    print inputDirs

    imageDir = os.path.dirname(listFile)
    #check files
    inputPathes = []
    for inputDir in inputDirs:
        imagePath = os.path.join(imageDir, inputDir, imageName)
        if not os.path.isfile(imagePath):
            print 'no file: ', imagePath
            exit(2)
        labelPath = os.path.join(imageDir, inputDir, labelName)
        if not os.path.isfile(labelPath):
            print 'no file: ', labelPath
            exit(2)
        maskPath = os.path.join(imageDir, inputDir, maskName)
        if not os.path.isfile(maskPath):
            print 'no file: ', maskPath
            exit(2)
        inputPathes.append([imagePath, labelPath, maskPath])

    for imagePath, labelPath, maskPath in inputPathes:
        #preprocess
        imagePreprocPath = imagePath+'_preproc.nrrd';
        labelPreprocPath = labelPath+'_preproc.nrrd';
        maskPreprocPath = maskPath+'_preproc.nrrd';

        args = [preprocessExe, '--input', imagePath.replace('/', os.sep), '--preset', 
        preset, '--binary', '0', '--spacing', (str(spacingXY)+' ')*3,
        '--output', imagePreprocPath.replace('/', os.sep)]
        ret = subprocess.call(args)
        if ret != 0:
            print 'error in ', str(args)
            exit(1)

        args = [preprocessExe, '--input', labelPath.replace('/', os.sep), '--preset',
        preset, '--binary', '1', '--spacing', (str(spacingXY)+' ')*3,
        '--output', labelPreprocPath.replace('/', os.sep)]
        ret = subprocess.call(args)
        if ret != 0:
            print 'error in ', str(args)
            exit(1)

        args = [preprocessExe, '--input', maskPath.replace('/', os.sep), '--preset',
        preset, '--binary', '1', '--spacing', (str(spacingXY)+' ')*3,
        '--output', maskPreprocPath.replace('/', os.sep)]
        ret = subprocess.call(args)
        if ret != 0:
            print 'error in ', str(args)
            exit(1)

        image = sitk.ReadImage(imagePreprocPath, sitk.sitkUInt8)
        label = sitk.ReadImage(labelPreprocPath, sitk.sitkUInt8)
        mask = sitk.ReadImage(maskPreprocPath, sitk.sitkUInt8)

        n = imagePath.split(os.path.sep)[-2]
        cutImageByTiles2p5d(image, label, mask, radius, stride, strideNegative, n, outputFolder, comment= '0.8')
    exit(0)