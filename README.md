# NeuralImageSynthesis
Code to reproduce the results from the paper "***Controlling Perceptual Factors in Neural Style Transfer***" (https://arxiv.org/abs/1611.07865).
The Jupyter Notebooks in the folder "ExampleNotebooks" give examples for all methods introduced in the paper. 
To get started, have a look at the BasicStyleTransfer.ipynb Notebook, which implements the standard Neural Style Transfer from *Image Style Transfer Using Convolutional Neural Networks (CVPR 2016)*.

To run the Notebooks you will need to download the pretrained VGG models:

`sh Models/download_models.sh`

The code to reproduce the control results for Fast Neural Style Transfer (Fig. 6) can be found at https://github.com/leongatys/fast-neural-style 

#Prerequisites
The most practical way to run the code is to use our jupyter-torch docker container. To get the container, use:

`docker pull bethgelab/jupyter-torch:cuda8.0-cudnn5`

To run the container, first get our docker toolchain: 
```
git pull https://github.com/bethgelab/docker
cd docker
```
Then from the docker folder run the following command, passing the arguments

1. GPU: comma separated list of the GPUs you want to access from the container
2. VOLUME: directory to be mounted in the container
3. XXXX: port at which the Jupyter Notebook is exposed on the Host
4. GROUPS: the groups the user in the docker container should be member of
5. USER: the user in the docker container
6. USER_ID: the user id in the docker container
7. CONTAINER_NAME: the name of the docker container
```
NV_GPU=GPU nvidia-docker run -v VOLUME:VOLUME -p XXXX:8888 -e USER_GROUPS=GROUPS -e USER=USER -e USER_ID=USER_ID -e USER_HOME=HOME -e GPU=GPU -d --name=CONTAINER_NAME bethgelab/jupyter-torch:cuda8.0-cudnn5
```
This should run the docker container. You can check if the container is running using

`docker ps`

Now, if you are on the Host, you can just open a browser and you should be able to access the Notebooks at localhost:XXXX

If you are remotely connecting to the Host, you can make an ssh tunnel, e.g. :

`ssh -f -N -L XXXX:localhost:XXXX Host`

Now you can open a browser on your local machine and access the Notebooks at localhost:XXXX.


#Disclaimer
This software is published for academic and non-commercial use only. 

This is research code, so you might find the documentation sometimes a bit wanting. 
Conceptually, what I wanted is to use Torch to evaluate any CNN activations and for the fast L-BFGS optimisation. 
At the same time, I wanted to work mainly in python using Jupyter Notebooks. 
That is why everything is done from the Notebooks and whenever there is need to compute CNN activations or run the optimisation, the inputs to the respective Torch script are saved as hdf5 files and they are called from the notebook. 
The Torch script then stores the results again in hdf5 files and they are pulled them back into the python notebook.
This might seem a little inefficient, but I found it to be a good flexibility vs. efficiency trade-off.

Finally, all this code was substantially inspired by the great implementation of our original Neural Style Transfer paper by Justin Johnson, which you can find at https://github.com/jcjohnson/neural-style.
