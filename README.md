# Bird Classification on Caltech-UCD dataset using CNNs

**Problem Statement :** 
​ ​Write code to implement simple ​ **bird image classification** ​ on ​ **customized
CNN** ​ The accuracy has to be ​ **above 95%** ​.You can use either ​ **TensorFlow** ​ or ​ **PyTorch.** ​You have to use the below dataset.

               [ http://www.vision.caltech.edu/visipedia/CUB-200.html​.](http://www.vision.caltech.edu/visipedia/CUB-200.html​.]

Download it, split into train, val, test sets in the ratio (70, 10, 20).

**Caltech-USD Birds 200** ​ : The dataset was released by Caltech
It's an image dataset with photos of 200 bird species (mostly North American). For detailed
information about the dataset, please see the technical report linked below.

   ● **Number of categories:** ​ 200

   ● **Number of images:** ​ 6033

   ● **Annotations:** ​ Bounding Box, Rough Segmentation, Attributes
Size of the images folder is 648 MB in .tgz format.

For this classification task we only need images and lists folder.

   ● Images

The images organized in subdirectories based on species.

   ● Lists

classes.txt : list of categories (species)

files.txt : list of all image files (including subdirectories)

train.txt : list of all images used for training

test.txt : list of all images used for testing

splits.mat : training/testing splits in MATLAB .mat format

Size after augmentation : 12066 Images (64X64)

Augmentation : Image Flip

  ● X_train : 16892

  ● X_test : 4826

  ● X_valid : 3016

Images were converted into numpy arrays and classes.txt folder was used to label the respective
images.


**CNN Architecture**
![image](https://user-images.githubusercontent.com/47039231/96599231-55ce0600-130d-11eb-8f95-bcb69332b3de.png)

**Brief Explanation :**

We start with 64 X 64 RGB (3 Channels) images and feed it to our 11 layer neural network.
The following diagrams will give an even better idea.

![image](https://user-images.githubusercontent.com/47039231/96599366-78601f00-130d-11eb-84e9-ba755d5386ef.png)

64@21X21 undergoes Max Pooling 3X3 to give 64@7X7 as shown in the next picture. Then followed by a
256@7X7 and another Max Pooling of 3X3 gives 256@2X2 which is flattened out into a 1024 vector.

![image](https://user-images.githubusercontent.com/47039231/96599524-a6456380-130d-11eb-8062-b83b71c2f0df.png)

Which undergoes matrix multiplication to give 512 vectors before undergoing another matrix multiplication
to give 201 numbers representing a dummy label and 200 species.To know more about the dummy label
go through the notebook.

![image](https://user-images.githubusercontent.com/47039231/96599639-c1b06e80-130d-11eb-8843-89782886a20d.png)


**Model Summary**

![image](https://user-images.githubusercontent.com/47039231/96599751-dd1b7980-130d-11eb-969f-bea51b72a81d.png)

**Parameters**

Library Used: Keras with Tensorflow

Loss Function: Multi-Label CrossEntropy

Optimizer Used : Adam

Learning Rate : Learning rate=0.001, Beta1=0.9, Beta2=0.

Dropout : 0.5 , that is half of total neurons but only during train time.

Total Number of epochs: 5X25 = 125


**Results**

Best Accuracy On Training Set : 90.61 %

Best Accuracy On Validation Set : 90.64 %

Best Accuracy On Test Set : 88.56 %

**Other Architecture Performances**
As the required accuracy was to be more than 95% I tried out a few more existing architectures
like Resnet18, Resnet 50, Vgg16 unfortunately I could not get any desired accuracy in fact lower ,
maybe because the dataset we were dealing with was too small for big architectures.
For these architecture's implementations fast.ai library was used.Data Augmentation was done
to avoid overfitting as much as possible using fast.ai Imagelist.
Results are as follows:
**Vgg16 with batchnorm​** : Maximum accuracy of 39% accuracy kept on oscillating back and forth
but did not move above 39% in any of the epochs.

![image](https://user-images.githubusercontent.com/47039231/96599930-09cf9100-130e-11eb-9f52-70454ff934f8.png)


**Resnet 18 :** ​ **​** It also did not perform well, got an mere accuracy of 20% and did not converge any
further,image size was changed to as described in paper to fit the model.

![image](https://user-images.githubusercontent.com/47039231/96600043-25d33280-130e-11eb-8871-a8d4fe5b0848.png)

**Resnet 50 :** 
​Resnet 50: It got an accuracy of about 70% , it did not improve any further. Since it’s a
much deeper network different types of augmentations were done like warping, random lightning
Unfortunately I do not have the performance picture


**Future Work**
  1. I think this performance could be increased by using 256X256 pixel inputs with a deeper model
than our current architecture but the problem availability of RAM in Google Colab the kernel
crashes when I tried to augment 256 X 256 pictures and convert them.

  2. I also tried without augmentation the RAM still is not sufficient and kernel crashes. So with a GPU Machine that has
a better RAM capacity we can expect better accuracy by carefully redesigning a better
architecture and applying good regularization techniques to avoid overfitting and get a better
generalization.

  3. I also tried with 128X128 it was not stable, that is it gave different results every time ,and best
result it produced was sometimes equal to the result produced by 64X

  4. Maybe Resnet 18 can also give a good result if we use more data by better augmentation and
pre-processing as given in paper.The reason we have got such a less accuracy may be because
of incorrect normalization.

  5. Whether Pretrained Network will improve accuracy is debatable because images in this dataset
overlap with images in ImageNet. We need to take extreme caution when using networks pre
trained as the test set of CUB may overlap with the training set of the original network.









 

