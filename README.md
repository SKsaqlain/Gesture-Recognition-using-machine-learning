# Gesture-Recognition-using-machine-learning
 The project was carried out as a part of Summer Internship At CDSAML, PES UNIVERSITY 2017.

## Working
<ul type=1>
 <li>SVM</li> 
<p>A Support Vector Machine (SVM) is a supervised machine learning algorithm that can be employed for both classification and regression purposes.
But in this case, it is used for classification purpose. SVMs are based on the idea of finding a hyperplane that best divides a data set into multiple classes. The SVM model was trained using hand digit images which were manually generated and processed, the dataset consists of 5 classes totally and each class consists of approximately ~20 images which were processed and resized into 25x25 dimensional image.<br/>
 RGB values of the hand were used to detect hand in the frame, once the hand was detected contours were constructed on the detected region to get the dimensions of the detected region so that the actual region can be extracted from the frame, once the region was extracted, the region of image was converted to binary after applying a threshold and then the entire region was raveled so that it becomes a single vector to be fed into the SVM model. The SVM model predicts the label for the input vector. The below gif gives a sample of the SVM models' output.
 
 ![](SVM_GIF.gif)
 </p>
 
 <li>DNN</li>
 <p>
 Neural networks are a set of algorithms, modeled loosely after the human brain, that is designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling or clustering raw input. The patterns they recognize are numerical, contained in vectors, into which all real-world data, be it images, sound, text or time series, must be translated. A DNN is an abstraction for Neural Networks such that it has an input layer, output layer, and multiple hidden layers.
 <br/>
  The DNN used in the model consists of total of 4 layers with ReLU as the activation function and softmax as the output function, the dataset that was used to train the model was MNIST dataset with 10 epochs and batch size to be 100 gave a corresponding accuracy of around ~97% ,the gestures corresponds to one of the digit pattern which on further classification was used to zoom into a region of an image.<br/>
 The corresponding digits/labels are used for the following purpose on correct identification.
 Digit 0 refers to the zoom in to the first quadrant, Digit 1 refers to zoom in to the second quadrant, Digit 2 refers to zoom in to the third quadrant, Digit 3 refers to zoom into a 4th quadrant and Digit 4 corresponds to move out to the original image. The gif showed below shows the sample output, press 'n' to clear the gesture area, press 'q' to predict and zoom in, press 'q' multiple times to quit.
 
 ![](Final_Sample_output.gif)
 </p>
 
 <li>CNN</li>
 <p>
  CNN's, like neural networks, are made up of neurons with learnable weights and biases. Each neuron receives several inputs, takes a weighted sum over them, pass it through an activation function and responds with an output. CNN's have wide applications in image and video recognition. Unlike neural networks, where the input is a vector, in DNN the input is an entire image that is a matrix where each convolutional layer is capable of learning a specific feature of the image. A CNN model was trained with 2 layers and ReLU as an activation function, the model was trained on the  MNIST dataset which on validating gave an accuracy of around ~95% but the model performed badly on real-time data as compared to the DNN model.
 </p>
</ul>



## Requirements
<ul type=1>
    <li>Python 3.6.5</li>
    <li>OpenCV 3</li>
    <li>Tensorflow 1.8.0 CPU support only</li> 
 </ul>
 
 
 ## Usage
To clone this repository.
```
$git clone https://github.com/SKsaqlain/Gesture-Recognition-using-machine-learning GRUML
```
Cd to  the Directory
```
$cd GRUML
```
To run the SVM model run the below commands, the sample output of shown in the Working section
```
$cd SVM MODEL
$python detect_SVM.py
```
To run the DNN model run the below commands form the parent directory i.e.,GRUML
```
$cd DNN model
$python DNN_FINAL_TEST_ZOOM_2.py
```
To run a pretrained model run the below command from the parent directory i.e.,GRUML
```
$cd TRAINED FINAL MODEL
$python model_save.py
```
The SIMPLE_APPLICATION folder consists of code where a haarcascade classifier is used to dected the fist<br/>
Move your fist to move the square cursor on the image correspondingly, show you plam to zoom into that region.<br/>
To run the code execute the below command from the parent directory.
```
$cd SIMPLE_APPLICATION
$python draw_1.py
```
