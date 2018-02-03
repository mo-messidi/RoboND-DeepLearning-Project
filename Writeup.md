# Semantic Segmentation Project

This project involves developing and training a fully convolutional network to do semantic segmentation of images from a drone camera stream to identify a target entity (class). Once the target it identified the drone can be instructed to perform a certain action. In this project, the drone is instructed to follow the identified target.

# Fully Convolutional Networks

FCNs are Convolutional Neural Networks (CNNs) that perverse the special information of images. They are comprised of three main building blocks.

## Encoding:
This part is very similar to traditional CNNs where an input image is passed through a series of convolutional layers, However, unlike in traditional CNNs, the fully connected layer that are used in traditional CNNs are replaced with 1x1 convolution (convolution with a kernel and stride of 1) layers. This maintains spatial image spatial information and avoids the image flattening that occurs when a fully connected layer is used.

## Decoding:
Decoding involves up-sampling the convoluted images that were previously encoded using a series of deconvolutional layers. This way, the output is an image that is the same size as the input image instead of a single ….  As is the case with traditional CNNs. For segmentation, each pixel is given assigned a label and the network tries to infer that label based on the labelled pixels in the training images.

## Skip connecting:
Utilize skip connections connect the deeper layers to shallow layers of the network. This is done to allow deeper layers to use the higher resolution information that is found in the shallow layers to give make more robust segmentation

# The network

![alt text](https://github.com/mo-messidi/RoboND-DeepLearning-Project/blob/master/code/model.png)

The FCN used in this project was consisted of 2 encoding blocks that use separable convolutions. The first convolution uses a filter = 32 and a stride = 2. The second convolution uses a filter = 64 and a stride = 2. The encoding section ends with a 1x1 convolution that uses a filter = 128. Two decoding blocks follow that use bilinear upsampling to deconvolute. The first de-convolution uses a filter = 64 and has a skip connection from the first encoder layer on the output side. The second de-convolution uses a filter = 32 and has a skip connection from the input layer on the output side.

Finally, after the second decoding block the output is passed to a softmax activation function to obtain the final model prediction.


All convolutions except the 1x1 convolution used same padding and a kernel size of 3x3. After each layer batch normalization was made to normalize the output and maintain the mean & variable of the data from each batch.

Separable convolutions were used to reduce the number of parameters needed and reduce the risk of overfitting. As opposed to a regular convolution where an output number of convulsions are performed on each channel of the input. In separable convolutions, a convolution is performed over each channel of an input layer and followed by a 1x1 convolution that takes the output channels from the previous step and then combines them into an output layer.

Bilinear upsampling uses the weighted average of the nearest known pixels from the given pixel to estimate the new pixel intensity value. Bilinear upsampling was chosen over transposed convolutions due to computational efficiency.

# Hyper parameters

A simple brute force trial method was used to select hyper parameters. Realistic performance limitations were also considered. The following parameter settings produced the best results:

learning_rate = 0.002

batch_size = 50

num_epochs = 100

steps_per_epoch = 200

validation_steps = 50

workers = 2

# Results

The best final score obtained was 0.44 with an IOU of 0.59.

# Future Enhancements

This model was trained based on 3 object classes (target, person and other). Including more object classes would result in better scene understanding assuming that training complexities are handled adequately. Also, a deeper network or one that is based on a tried and prove architecture like VGG may provide better results if used in the future.
