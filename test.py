# Code for the estimation of Photoz and associated PDFs
# paper : "Photometric redshifts from SDSS images using a Convolutional Neural Network" 
# authors : J.Pasquet, E. Bertin, M. Treyer, S. Arnouts and D. Fouchez

import sys
import numpy as np
import os
from network import *
import matplotlib.pyplot as plt


##############PARAMETERS#################

NB_BINS=60*3
BATCH_SIZE=64
ZMIN=0.0
ZMAX=0.4
BIN_SIZE=(ZMAX-ZMIN)/NB_BINS
range_z=np.linspace(ZMIN,ZMAX,NB_BINS+1)[:NB_BINS]

#############LOADING DATA########
npz_test = "data/data_example.npz"
path_model="pretrained_model/"


set_cube = np.load(npz_test)
data = set_cube["data"]
z = set_cube["z"]
ebv = set_cube["ebv"]

print data.shape
if not(data.shape[1]==64 and data.shape[2]==64 and data.shape[3]==5):
	print "The datacube has to have a dimension of Nx64x64x5!"
	exit(1)


#############LOADING WEIGHTS########
params = model()
session_conf = tf.ConfigProto()
session_conf.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=session_conf)
saver = tf.train.Saver()
saver.restore(session,tf.train.latest_checkpoint(path_model))

prediction=[]
i=0

for i in range(0,data.shape[0],BATCH_SIZE):
	batch_data=data[i:min(i+BATCH_SIZE,data.shape[0])]
	batch_ebv=ebv[i:min(i+BATCH_SIZE,ebv.shape[0])]

	dico={params["x"]:batch_data,params["reddening"]:batch_ebv}
	output=session.run(params["output"],feed_dict=dico)

	tmp= np.sum(output*range_z,axis=1)
	prediction = prediction+list(tmp)



prediction=np.array(prediction)

z=z[:,0]

err_abs =  np.sum( abs(prediction-z ))/z.shape[0]
deltaz=(prediction-z)/(1+z)
biais=np.sum(deltaz)/z.shape[0]
nmad= 1.48*np.median(abs(deltaz-np.median(deltaz)))
print "biais = ", " sigma_mad = ",nmad

