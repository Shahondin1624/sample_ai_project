Using the [MNIST-Dataset](https://pjreddie.com/projects/mnist-in-csv/) I try to determine which
combination of 
- epochs 
- hidden nodes 
- learning rate

yields the best performing network. My knowledge in Python is fairly limited, so my code may
not be pythonic and certainly is very un-optimized.

For this script to work, there needs to be a "samples"-folder containing the test and training
dataset named ("mnist_test.csv" and "mnist_train.csv") in the project folder.
Furthermore a folder "models" is required to save the resulting models.