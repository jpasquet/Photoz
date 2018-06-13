import numpy as np
import tensorflow as tf


def prelu(x):
	with tf.name_scope('PRELU'):
		_alpha = tf.get_variable("prelu", shape=x.get_shape()[-1],
		dtype=x.dtype, initializer=tf.constant_initializer(0.0))


	return tf.maximum(0.0, x) + _alpha * tf.minimum(0.0, x)


def conv2d(input,
	   num_output_channels,
	   kernel_size,name):

	with tf.variable_scope(name):


		num_in_channels = input.get_shape()[-1].value
		kernel_shape = [kernel_size, kernel_size, num_in_channels, num_output_channels]

		biases = tf.get_variable('biases', shape=[num_output_channels], initializer=tf.constant_initializer(0.1))
		kernel =  tf.get_variable('weights',  shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer())

		outputs = tf.nn.conv2d(input, kernel,strides=[1,1,1,1], padding="SAME")
		outputs = tf.nn.bias_add(outputs, biases)		
		outputs = prelu(outputs)

	return outputs

def pool2d(input,kernel_size,stride,name):
	print input, [1, kernel_size, kernel_size, 1],[1, stride, stride], 1
	with tf.variable_scope(name):
		return tf.nn.avg_pool(input, ksize=[1, kernel_size, kernel_size, 1],strides=[1, stride, stride, 1],padding="SAME",name=name)


def fully_connected(input, num_outputs, name, withrelu=True):
		           
	with tf.variable_scope(name):
		num_input_units = input.get_shape()[-1].value
		kernel_shape=[num_input_units, num_outputs]
		kernel =  tf.get_variable('weights',  shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer())

		outputs = tf.matmul(input, kernel)
		biases = tf.get_variable('biases', shape=[num_outputs], initializer=tf.constant_initializer(0.1))
		
		outputs = tf.nn.bias_add(outputs, biases)
		if withrelu:
			outputs = tf.nn.relu(outputs)

		return outputs




def inception(input,nbS1,nbS2,name,output_name,without_kernel_5=False):
	with tf.variable_scope(name):


		s1_0=conv2d(input=input,
				num_output_channels=nbS1,
				kernel_size=1,
				name=name+"S1_0")

		s2_0=conv2d(input=s1_0,
		   num_output_channels=nbS2,
		   kernel_size=3,
		   name=name+"S2_0" )



		s1_2=conv2d(input=input,
		   num_output_channels=nbS1,
		   kernel_size=1,
		   name=name+"S1_2")
		 
		pool0=pool2d(input=s1_2,
				kernel_size=2,
				stride=1,
				name=name+"pool0")


		if not(without_kernel_5):
			s1_1=conv2d(input=input,
			   num_output_channels=nbS1,
			   kernel_size=1,
			   name=name+"S1_1")

			s2_1=conv2d(input=s1_1,
			   num_output_channels=nbS2,
			   kernel_size=5,
			   name=name+"S2_1")

		
		s2_2=conv2d(input=input,
		   num_output_channels=nbS2,
		   kernel_size=1,
		   name=name+"S2_2")
		  


		if not(without_kernel_5):
			output=tf.concat(values =[s2_2,s2_1,s2_0,pool0], name=output_name,    axis=3)
		else:
			output =tf.concat(values=[s2_2,s2_0,pool0], name=output_name,    axis=3)

	return output




def model():


	reddening= tf.placeholder(tf.float32, shape=[None,1], name="reddening")
	x= tf.placeholder(tf.float32, shape=[None,64,64,5], name="x")


	conv0 = conv2d(input=x,
		num_output_channels=64,
		kernel_size=5,
		name="conv0")

	conv0p = pool2d(input=conv0,
			kernel_size=2,
			stride=2,
			name="conv0p")
			


	i0 = inception(conv0p,48,64,name="I0_", output_name="INCEPTION0")
	i1 =inception(i0,64,92,name="I1_", output_name="INCEPTION1")

	i1p=pool2d(input=i1,
			kernel_size=2,
			name="INCEPTION1p",
			stride=2)


	i2=inception(i1p,92,128,name="I2_", output_name="INCEPTION2")
	i3=inception(i2,92,128,name="I3_", output_name="INCEPTION3")

	i3p=pool2d(input=i3,
			kernel_size=2,
			name="INCEPTION3p",
			stride=2)


	i4=inception(i3p,92,128,name="I4_", output_name="INCEPTION4", without_kernel_5=True)


	flat = tf.layers.Flatten()(i4)


	concat =tf.concat(values=[flat,reddening],axis=1)



	fc0 = fully_connected(input=concat,
				num_outputs=1096,
				name="fc0")

	fc1 = fully_connected(input=fc0,
				num_outputs=1096,
				name="fc0b")
				

	fc2 = fully_connected(input=fc1,
				num_outputs=180,
				name="fc1", withrelu=False)
	
	output=tf.nn.softmax(fc2)

	params={"output":output,"x":x,"reddening":reddening}
	return params
