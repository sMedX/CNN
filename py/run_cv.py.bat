set "path=%path%;D:\apb\distributions\win64\Autoplan\bin;D:\apb\SB\Autoplan-build\bin\data\modules\autoplansegmentationtools\caffe;D:\alex\caffe-ms\NugetPackages\gflags.2.1.2.1\build\native\x64\v120\dynamic\Lib"

set "preproc=D:\alex\CNN-utils-build\Release\runPreprocess.exe"
rem set "cut=D:/alex/CNN-utils-build/Release/CutImageByTiles.exe"
set "cut=D:/alex/CNN/py/CutImageByTiles.py"
set "caffe=D:/alex/ms-caffe-rep-copy1/Build/x64/Release/caffe.exe"
set "clas=D:/alex/CNN-build4/classification/Release/classification.exe"
set "postproc=D:/alex/CNN-utils-build/Release/postprocessing.exe"
set "valid=D:/alex/CNN-utils-build/Release/Validation.exe"
set "python=python.exe"

set "nets=D:/alex/caffe-nets"
set "images=D:/alex/images"
set "tiles=D:/alex/tiles"
set "snaps=D:/alex/caffe-nets/snaps"

%python% run_cv.py
pause