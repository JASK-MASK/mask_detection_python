# Mask detector using CNN

You must follow some steps in order to make your mask detector work. First of all, you need
to install the needed libraries, for this I already uploaded a .txt file which contains all the
necessary libraries to use with your system. This program was coded in GNU/Linux, so it may be better
to use this OS to avoid any kind of problems. So, once in terminal you should command: "pip/pip3 install -r install_libs.txt" or if you work with anaconda
environments you should write: "conda install -r install_libs.txt"

This will take a while. You can check out if you got all the libraries installed correctly if you introduce python (or python3) in your command line,
after this you can try to import every single library to see if it works. By any chance you have an error, you should install the library individually.

After this, go ahead and run the detect_video.ipynb file and check the results. If you have any doubts, questions or concerns, you can contact me at
alejandro.zunigaperez@ucr.ac.cr, and I will be more than glad to help you out.

I will really appreciate if there appear some helpful critics about how to improve the program and work with it efficiently.

# Notes: 

1. DO NOT run the train_model.ipynb file, it will take several hours to execute. The purpose of this code is to demonstrate I created my own model
and if you open it you can see every jupyter cell executed correctly, obtaining a good accuracy for the dataset used. 

2. The dataset used is an open source file which was used for preprocessing data and training the model. You can use your own dataset but probably you
will get either better or worse results. In this case you actually need to re-run your own classification model.

3. If you use Linux in a Virtual Machine (was my personal case), you need to install VirtualBox extensions in order to work with your camera,
otherwise you will not work with the detector.

4. Be sure you use download or clone the repository in a same file or directory, otherwise you need to go ahead and change the path for each opened file.

5. I was inspired on Adrian Rosenbrocks course at https://www.pyimagesearch.com/
