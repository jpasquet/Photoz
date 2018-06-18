#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Example code for estimating photo-zs and the associated PDFs from ugriz images
# From "Photometric redshifts from SDSS images using a Convolutional Neural Network" 
# by J.Pasquet et al. 2018

import sys
import numpy as np
import os
from network import *
import matplotlib.pyplot as plt


##############PARAMETERS#################

NB_BINS = 60 * 3
BATCH_SIZE = 64
ZMIN = 0.0
ZMAX = 0.4
BIN_SIZE = (ZMAX - ZMIN) / NB_BINS
range_z = np.linspace(ZMIN, ZMAX, NB_BINS + 1)[:NB_BINS]

#############LOAD DATA########
npz_test = "data/data_example.npz"
path_model = "pretrained_model/"

set_cube = np.load(npz_test)
data = set_cube["data"]
z = set_cube["z"]
ebv = set_cube["ebv"]

print(data.shape)
if not(data.shape[1] == 64 and data.shape[2] == 64 and data.shape[3] == 5):
	print("The shape of the datacube must be Nx64x64x5!")
	exit(1)

#############RESTORE NN WEIGHTS########
params = model()
session_conf = tf.ConfigProto()
session_conf.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=session_conf)
saver = tf.train.Saver()
saver.restore(session,tf.train.latest_checkpoint(path_model))

############FETCH THE NN USING MINIBATCHES############
prediction = []
probas=[]
for i in range(0, data.shape[0], BATCH_SIZE):
	batch_data = data[i : min(i + BATCH_SIZE, data.shape[0])]
	batch_ebv = ebv[i : min(i + BATCH_SIZE, ebv.shape[0])]
	dico = {params["x"]:batch_data,params["reddening"]:batch_ebv}
	output = session.run(params["output"], feed_dict=dico)
	probas = probas +list(output)
	prediction = prediction + list(np.sum(output * range_z, axis=1))

probas=np.array(probas)
prediction = np.array(prediction)
z = z[:,0]

############SHOW STATISTICS############
err_abs = np.sum(abs(prediction - z)) / z.shape[0]
deltaz = (prediction - z) / (1 + z)
bias = np.sum(deltaz) / z.shape[0]
nmad = 1.48 * np.median(abs(deltaz - np.median(deltaz)))
print(" N = %d galaxies" %z.size)
print(" bias = %.4g" %bias)
print(" sigma_mad = %.4g" %nmad)


############SAVE FILE############
tmp = np.zeros((z.shape[0], probas.shape[1]+2))
tmp[:, 0] = z
tmp[:, 1] = prediction
tmp[:, 2:] = probas
np.savetxt("output.txt",tmp, fmt=["%1.4f", "%1.4f"]+["%1.8f"]*probas.shape[1], header="z[1.4f] zphot_CNN[1.4f] probas[1.8f]x180")
