GAN/mnist_dcgan.py : Code for deep GAN
GAN/mnist_gan.py : Code for GAN
convnetFinalCode.py : Code for training a CNN network to work on MNIST dataset
capnetFinalCode.py: Code for training CapNet on MNIST dataset
imageRead.py: Used to create overlapping digits
label.py: Used to label the DCGAN generated images

overlapped_digits/: contains samples of overlapped MNIST images
gan_dataset_resized/: contains dcgan generated image sample

CapNet_MNIST_FASHION/main.py : python script for training the capnet
Training
$ python main.py
$ # or training for fashion-mnist dataset
$ python main.py --dataset fashion-mnist

Calculate test accuracy
$ python main.py --is_training=False
$ # for fashion-mnist dataset
$ python main.py --dataset fashion-mnist --is_training=False

