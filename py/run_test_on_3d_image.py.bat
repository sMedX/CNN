set "path=%path%;D:\apb\distributions\win64\Autoplan\bin;D:\alex\CNN-build4\Release;D:\apb\distributions\win64\Autoplan\bin\data\modules\autoplansegmentationtools\caffe"

set "clas=D:/alex/CNN-build4/classification/Release/classification.exe"
set "postproc=D:/alex/CNN-utils-build/Release/postprocessing.exe"
set "valid=D:/alex/CNN-utils-build/Release/Validation.exe"
set "python=python.exe"

set "images=D:/alex/images"

%python% run_test_on_3d_image.py

pause