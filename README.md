# BioSonar-IC-model
Echolocating bats have a specialized auditory system to navigate complex environments for huning, navigation and survival.
This code is to build a biomimetic network which emulates auditory characteristics of the big brown bat's IC neurons.

# Development environment
This code was developed in the following condition.
  - CentOS Linux 7
  - Python 2.7
  - tensorflow 1.15.0
  - h5py 2.10.0
  - numpy 1.16.6

# Usage of this code
  - Generate a set of auditory-spectrograms [1]\
    Ex) On MATLAB, Variable name: Data\
        Data = reshape(Data, [# of audio samples, 128 * nFrames]);\
        save('./FILENAME.mat', 'Data', '-v7.3');
  - Copy the mat file to the directory "./Data/FILENAME.mat"
  - Modify two parameters, nTrnData and nValData.\
    Ex) Usually, 10 % of data is used for validation
  - Run, python train_mono_stacking_model.py
    
# Reference
  [1] Chi T, Shamma S. NSL Matlab Toolbox. University of Maryland, Colleage Park. 2005. 
