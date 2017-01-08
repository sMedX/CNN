#export PATH=$PATH:/home/himanshu/practice/

#todo
export preproc=/root/host/CNN-utils-build/runPreprocess
#export cut=/root/host/CNN-utils-build/cutImageByTiles
export cut=/root/host/CNN/py/cutImageByTiles.py
export caffe=caffe
export clas=
export postproc=
export valid=
export python=python

export nets=/root/host/caffe_nets
export images=/root/host/images
export tiles=/root/host/tiles_new
export snaps=/root/host/caffe_nets/snaps

$python run_cv.py
