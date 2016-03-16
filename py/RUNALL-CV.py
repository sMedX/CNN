import os
import subprocess
from subprocess import call
import errno

classCount = '2'
cut = 'D:\\alex\\CNN-build\\utils\\Release\\CutImageByTiles.exe'
makeTileLists = ['python.exe', 'make_sample_names_' + classCount + '-classes.py']
train = 'C:\\caffe\\bin\\caffe_cc35.exe'
clas = 'D:\\alex\\nets_pancreas\\caffe-win-1029\\bin\\class_new.exe'
postproc = 'D:\\alex\\CNN-build\\postprocessing\\Release\\postprocessing.exe'
valid = 'D:\\alex\\CNN-build\\validation\\Release\\Validation.exe'

preset = 'livertumors'
ver = '25'
spacing = '0.782'
spacingStr = 'orig' if spacing=='0' else spacing

deviceId = '1'
iters = '30000'

dir=os.path.join('C:/caffe', preset, ver)
print dir

r = '32'
size = int(r) * 2
label = preset + '.nrrd'
mask = 'liver.nrrd'
patient = 'patient.nrrd'
  
imagesPath = 'D:\\alex\\images'
tilesFolder = os.path.join('D:\\alex\\tiles', preset, str(size) + 'x' + str(size), 'sampling-' + spacingStr)
samplesList = os.path.join(imagesPath, preset, 'samplesListGood.txt')

deploy = os.path.join('C:\\caffe', preset, ver, 'deploy.prototxt')
start = [0, 0, 0]
size = [512, 512, 1000]
batchLength = '1024'
groupX = '3'
groupY = '3'
print preset,',v-', ver, ',spacing-', spacing

snaphotPrefix='D:\\Artem\\caffe\\snap'
sigma = '4.000000'

def main():

    #retcode = subprocess.call([cut, '-listFile', samplesList, '-imageName', patient, '-labelName1', label,
    #                 '-labelName2', 'livertumors_dark.nrrd', '-maskName', mask, '-radius', r, '-preset', preset,
    #                 '-stride 0 0 0', '-spacingXY', spacing, spacing, '-strideNegative 1', '-folder', tilesFolder])
    #
    #if (retcode != 0):
    #    print 'error. ', cut, ' exit with ', retcode
    #    return

    n = 4  # number of parts for cross-validation

    for k in range(0, n):
        kn = str(k) + '-' + str(n)
        # create folder for snapshots
        snapshotFolder = os.path.join(snaphotPrefix, preset, ver, kn)
        try:
            os.makedirs(snapshotFolder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise e
            pass
                
        model = os.path.join(snapshotFolder, '_iter_' + iters + '.caffemodel')

        # make lists
        retcode = subprocess.call([makeTileLists[0], makeTileLists[1], tilesFolder, dir, imagesPath, samplesList, str(k), str(n)])
        if retcode!=0:
            print 'error. ', cut, ' exit with ', retcode
            return
            
        sampleTrainListK = os.path.join(dir, 'train-cv-' + kn + '.txt')
        sampleTestListK = os.path.join(dir, 'test-cv-' + kn + '.txt')

        tileTrainListK = os.path.join(dir, 'tileList-train-cv-' + kn + '.txt')
        tileTestListK = os.path.join(dir, 'tileList-test-cv-' + kn + '.txt')

        # make net and solver
        solver = createNetAndSolver(kn, tileTestListK, tileTrainListK, iters, snapshotFolder)
        
        subprocess.call([train, 'train', '--solver', solver, '--gpu', deviceId])

        suffix = '-v' + ver + '-g' + groupX + 'x' + groupY + '-c' + classCount + '-s' + spacing + '-cv' + kn + '.nrrd'
        outputCNN = preset + '-cnn' + suffix

        with open(samplesList) as f:
            for line in f:
                line=line.replace('\n','')
                args=[clas, deploy, model, str(start[0]), str(start[1]), str(start[2]), str(size[0]), str(size[1]), str(size[2]), r,
                preset, spacing, batchLength, groupX, groupY, classCount, os.path.join(line, patient),
                os.path.join(line, mask), os.path.join(line, outputCNN), deviceId]
                print args
                subprocess.call(args)
        
        #validate(sampleTrainListK, sampleTestListK, label, outputCNN, suffix)          

        with open(samplesList) as f:
            for line in f:
                line = line.replace('\n','')
                subprocess.call([postproc, '-image', os.path.join(line, outputCNN), '-gaussianVariance', sigma])
        
        suffix = suffix + '-gaussian' + sigma + '.nrrd'
        outputCNN = preset + '-cnn' + suffix

        validate(sampleTrainListK, sampleTestListK, label, outputCNN, suffix)
    return


def validate(samplesTrainListK, samplesTestListK, label, outputCNN, suffix):
    with open(samplesTrainListK) as f:
        for line in f:
            line = line.replace('\n','')
            subprocess.call([valid, '-testImage', os.path.join(line, outputCNN), '-label', os.path.join(line, label),
                             '-report', 'report-cnn-train-' + preset + '-' + suffix + '.csv'])
    with open(samplesTestListK) as f:
        for line in f:
            line = line.replace('\n','')
            subprocess.call([valid, '-testImage', os.path.join(line, outputCNN), '-label', os.path.join(line, label),
                             '-report', 'report-cnn-test-'  + preset + '-'+ suffix + '.csv'])
    return valid


def createNetAndSolver(kn, tileTestListK, tileTrainListK, iters, snapshotFolder):
    netTemplate = os.path.join(dir, 'netTemplate.prototxt')
    net = os.path.join(dir, 'net-cv-' + kn + '.prototxt')
    fileData = None
    with open(netTemplate, 'r') as file:
        fileData = file.read()
    fileData = fileData \
        .replace('%trainTilesList%', tileTrainListK.replace('\\','/')) \
        .replace('%testTilesList%', tileTestListK.replace('\\','/'))
    with open(net, 'w') as file:
        file.write(fileData)
    print 'net ' + net + ' has been created'
    
    solverTemplate = os.path.join(dir, 'solverTemplate.prototxt')
    solver = os.path.join(dir, 'solver-cv-' + kn + '.prototxt')
    fileData = None
    with open(solverTemplate, 'r') as file:
        fileData = file.read()
    fileData = fileData \
        .replace('%net%', net.replace('\\','/')) \
        .replace('%iters%', iters) \
        .replace('%snapshotFolder%', snapshotFolder.replace('\\','/')+'/')
    with open(solver, 'w') as file:
        file.write(fileData)
    print 'solver ' + solver + ' has been created'

    return solver

if __name__ == "__main__":
    main()