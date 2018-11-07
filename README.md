# Simple Feedforward Neural Network With Tensors

C++ code for a modular, simple, feed forward neural network library

Code to go along with blog: [Tensors, The Building Block For Deep Learning](http://www.curiousinspiration.com/posts/tensors,-the-building-block-for-deep-learning)

# Build

`mkdir build`

`cd build`

`cmake -D GLOG_INCLUDE_DIR=~/Code/3rdParty/glog-0.3.5/glog-install/include/ \
      -D GLOG_LIB_DIR=~/Code/3rdParty/glog-0.3.5/glog-install/lib/ \
      -D GTEST_INCLUDE_DIR=~/Code/3rdParty/googletest-release-1.8.0/install/include/ \
      -D GTEST_LIB_DIR=~/Code/3rdParty/googletest-release-1.8.0/install/lib/ ..`

`make`

`./tests`

`./feedforward_neural_net`