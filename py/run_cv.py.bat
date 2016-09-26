set "path=%path%;D:\apb\SB\Autoplan-build\bin\data\modules\autoplansegmentationtools\caffe;D:\alex\caffe-ms\NugetPackages\gflags.2.1.2.1\build\native\x64\v120\dynamic\Lib"

set "cut=D:/alex/CNN-build4/utils/Release/CutImageByTiles.exe"
set "caffe=D:/alex/ms-caffe-rep-copy1/Build/x64/Release/caffe.exe"
set "clas=D:/alex/CNN-build4/classification/Release/classification.exe"
set "postproc=D:/alex/CNN-build4/postprocessing/Release/postprocessing.exe"
set "valid=D:/alex/CNN-build4/validation/Release/Validation.exe"
set "python=python.exe"

set "nets=D:/alex/caffe-nets"
set "images=D:/alex/images"
set "tiles=D:/alex/tiles"
set "snaps=D:/Artem/caffe/snap"

%python% run_cv.py