This is a mini-project for the robotic perception subject.

The project tries to implement a detection software that can classify a series of object. In this case the category of objects is Instruments where in four classes have been defined, Acoustic Guitar, electric bass, trumpet and a drumm set.
This mini-project is very simple and therefore each object is displayed on a white background, making image processing simple.


There are three executable python files in this project
1. instrument_data.py
2. k_nearest_detector.py
3. bayers_class_detector.py

The data file is used to generate data from various training images.
k_nearest detector is the detection program using k-nearest as the classifier, while bayeres_class uses Bayes instead 

To test with your own images simple put .jpg files into the "materialer" folder and assign them a number, the amount of pictures tested can then be changed in the instrument_detector file