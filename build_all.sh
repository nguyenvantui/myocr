#compile_nms
!python setup.py build develop

# compile roi_pooling
%cd model/roi_pool/
!python setup.py build_ext --inplace
