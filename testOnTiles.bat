rem C:\caffe\bin\caffe_cc35.exe old caffe

set preset=livertumors
set ver=27
C:
cd caffe/%preset%/%ver%

D:\alex\caffe-ms\caffe-master\Build\x64\Release\caffe.exe test -model deploy_test.prototxt -weights D:/Artem/caffe/snap/%preset%/%ver%/_iter_40000.caffemodel -gpu 1 -iterations 1200
