# Photometric redshifts from images using a Convolutional Neural Network

This repository contains the [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) model as well as a subset of multispectral [SDSS](http://www.sdss.org/) images for testing.

Checkout [J. Pasquet et al. 2018](http://arxiv.org/) for a detailed description of the model and its performance.

The `test.py` [Python](https://www.python.org/)  code runs the CNN in inference mode on 100 example images (stored in `data/data_example.npz`) using a set of pretrained weights. The code computes a [photometric redshift](https://en.wikipedia.org/wiki/Photometric_redshift) estimate and the associated [PDF](https://en.wikipedia.org/wiki/Probability_density_function) over 180 redshift bins for each of the provided galaxy images.  

The code has been tested with Python 2.7 and Python 3.6. The CNN model is built on top of the [TensorFlow](https://www.tensorflow.org/) framework and should be compatible with TF versions ≥ 1.4.1.

**Installing**

[Git](https://git-scm.com/) and [Git-LFS](https://git-lfs.github.com/) are required for cloning the present repository (the CNN weights exceed GitHub's regular 100MB filesize limit). Alternatively you may download the pretrained_model data [from here](https://drive.google.com/drive/folders/19QjIaJcbe7btlUDTHUWxC64-aEQk4r9Q).

On UN*X systems:
```
pip install --upgrade matplotlib numpy tensorflow
git clone https://github.com/jpasquet/photoz
cd photoz
git lfs install
git lfs pull
```

**Running the code**
```
python test.py
```
