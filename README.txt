####README####
UPDATE: Please go to the colab notebook from this link https://colab.research.google.com/drive/1s49zHKAiSX9pd3pEq4R0iyzhCFaJcSCe?usp=sharing
Packages:
If you are using Google Colaboratory then you do not need to install any packages because all the packages are pre installed you are only required to 
import them which will be step 2, but if you are using local machine then these are the packages you have to install.
Open Python type the following commands to install the respective package.  
Keras     : pip install keras
Tensorflow: pip3 install --user --upgrade tensorflow
glob is preinstalled in python
OpenCV:
Download latest OpenCV release from sourceforge site and double-click to extract it.
Goto opencv/build/python/2.7 folder.
Copy cv2. pyd to C:/Python27/lib/site-packeges.
Open Python IDLE and type following codes in Python terminal. >>> import cv2 >>> print cv2. __version__
numpy,matplotlib are pre installed in python
After installing all the packages you can import them easily, for detailed imports please go through .ipynb/.py file.



Datatset:


 Images:http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz  (648MB .tgz file)
 Lists :http://www.vision.caltech.edu/visipedia-data/CUB-200/lists.tgz   

Caltech serves are a bit slow it will roughly take more than a hour on a good internet connection, I have already downloaded the dataset 
and stored it in my drive under the name 'UCSD 200'. You can use it as well.Add it to your drive

  Link : https://drive.google.com/drive/folders/1Sq0HVOXwE5fr-jR0iSoT-wjJsnIX0mxR?usp=sharing
 
Add it your drive account and follow the following instructions.



Instructions

After Placing  the Dataset in your Google drive,open colabarotry, mount your drive and  execute the 'Import Block' followed by 'Extracting All Files' 
Codeblock from my ipynb/py file.
The Notebook now has different blocks making it flexible to execute. Each block as a seperate funciton.Execute each block one by one or independently
the other blocks.Keep in mind some blocks have depenence on other blocks.
If you want to check the perfomance of the smodel follow this procedure:

Load the deleiverables.zip in to your Drive 
Link : https://drive.google.com/file/d/1i3mV36aeF4cPCwzcCnY2qPvglRg_i8Lq/view?ts=5e707e7b
deleiverables.zip(84.4MB) folder has the following:
 Best Model(best_model.h5)
 Best Model Weights (best_weights.h5)
 X_test,y_test(X_test.npy,y_test.npy)

Execute the Import Block, Custom CNN Block which contains the 'create model function' and unzip deleiverables.zip

The code is as follows 

!unzip deleiverables.zip
X_test=np.load('X_test.npy')
y_test=np.load('y_test.npy')
model=create_model()
model.load_weights('best_weights.h5')

Now you can execute this code on Evaluation  Block for checking the model's perfomance.

For Detailed Report about Architecture please read my report and go through the notebook.



 

