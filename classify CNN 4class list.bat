
set "exe=D:\alex\nets_pancreas\caffe-win-1029\bin\class_new.exe"
set "deploy=C:\caffe\new_tumors\newtum06\newtum_06_deploy1024_4classes.prototxt"
rem set "model=D:/artem/caffe/snap/newtum_08/newtum_08_iter_430000.caffemodel"
rem set "model=D:/artem/caffe/snap/newtum_08_1/newtum_08_iter_235000.caffemodel"
set "model=D:\Artem\caffe\snap\newtum_09\newtum_09_iter_450000.caffemodel"
set "start=0 0 0"
set "size=511 511 1000"
set "r=32"
set "preset=livertumors"
set "spacing=0.782"
set "batchLength=1024"
set "groupX=3"
set "groupY=3"
set "classCount=4"
set "deviceId=1"

FOR /F %%G IN (D:\alex\images\samplesList2.txt) DO %exe% %deploy% %model% %start% %size% %r% %preset% %spacing% %batchLength% %groupX% %groupY% %classCount% %%G\patient.nrrd %%G\liver.nrrd %%G\livertumors-cnn-4ada.nrrd %deviceId%

pause