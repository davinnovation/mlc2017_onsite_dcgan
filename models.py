# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf
import utils

import tensorflow.contrib.slim as slim

def xavier_init(size):
  in_dim = size[0]
  xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
  return tf.random_normal(shape=size, stddev=xavier_stddev)

"""Contains the base class for models."""
class BaseModel(object):
  """Inherit from this class when implementing new models."""

  def create_model(self, **unused_params):
    """Define variables of the model."""
    raise NotImplementedError()

  def run_model(self, unused_model_input, **unused_params):
    """Run model with given input."""
    raise NotImplementedError()

  def get_variables(self):
    """Return all variables used by the model for training."""
    raise NotImplementedError()

class SampleGenerator(BaseModel):
  def __init__(self):
    self.noise_input_size = 100

  def create_model(self, output_size, **unused_params):
    h1_size = 128
    self.G_W1 = tf.Variable(xavier_init([self.noise_input_size, h1_size]), name='g/w1')
    self.G_b1 = tf.Variable(tf.zeros(shape=[h1_size]), name='g/b1')

    self.G_W2 = tf.Variable(xavier_init([h1_size, output_size]), name='g/w2')
    self.G_b2 = tf.Variable(tf.zeros(shape=[output_size]), name='g/b2')

  def run_model(self, model_input, is_training=True, **unused_params):
    net = tf.nn.relu(tf.matmul(model_input, self.G_W1) + self.G_b1)
    output = tf.nn.sigmoid(tf.matmul(net, self.G_W2) + self.G_b2)
    return {"output": output}

  def get_variables(self):
    return [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

class SampleDiscriminator(BaseModel):
  def create_model(self, input_size, **unused_params):
    h1_size = 128
    self.D_W1 = tf.Variable(xavier_init([input_size, h1_size]), name='d/w1')
    self.D_b1 = tf.Variable(tf.zeros(shape=[h1_size]), name='d/b1')

    self.D_W2 = tf.Variable(xavier_init([h1_size, 1]), name='d/w2')
    self.D_b2 = tf.Variable(tf.zeros(shape=[1]), name='d/b2')

  def run_model(self, model_input, is_training=True, **unused_params):
    net = tf.nn.relu(tf.matmul(model_input, self.D_W1) + self.D_b1)
    logits = tf.matmul(net, self.D_W2) + self.D_b2
    predictions = tf.nn.sigmoid(logits)
    return {"logits": logits, "predictions": predictions}

  def get_variables(self):
    return [self.D_W1, self.D_W2, self.D_b1, self.D_b2]

class DCGAN_Gen(BaseModel):
  def create_model(self, input_size, **unused_params):
    self.noise_input_size = 100    
    # linear layer
    h1_size = 4*4*512
    self.G_W1 = tf.Variable(xavier_init([self.noise_input_size, h1_size]), name='g/w1')
    self.G_b1 = tf.Variable(tf.zeros(shape=[h1_size]), name='g/b1')
    
    #batch_norm layer
    self.bn1_beta = tf.Variable(tf.zeros([512]),name='g/bn1')
    self.bn1_scale = tf.Variable(tf.ones([512]),name='g/bn1_s')
    self.bn2_beta = tf.Variable(tf.zeros([256]),name='g/bn2')
    self.bn2_scale = tf.Variable(tf.ones([256]),name='g/bn2_s')
    self.bn3_beta = tf.Variable(tf.zeros([128]),name='g/bn3')
    self.bn3_scale = tf.Variable(tf.ones([128]),name='g/bn3_s')    
    #conv layer
    self.dc1 = tf.Variable(tf.random_normal([5,5,256,512],stddev=0.01),name='g/dc1')
    self.dc1_b = tf.Variable(tf.zeros([256]),name='g/dc1_b')
    self.dc2 = tf.Variable(tf.random_normal([5,5,128,256],stddev=0.01),name='g/dc2') 
    self.dc2_b = tf.Variable(tf.zeros([128]),name='g/dc2_b')
    self.dc5 = tf.Variable(tf.random_normal([5,5,1,128],stddev=0.01),name='g/dc3')
    self.dc5_b = tf.Variable(tf.zeros([1]),name='g/dn3_b')

    
  def run_model(self, model_input, is_training=True, **unused_params):
    print(model_input.shape, "genshape")
    batch_size = tf.shape(model_input)[0]
    alpha = 0.2
    epsilon = 1e-3
    x1 = tf.matmul(model_input, self.G_W1)
    x1 = tf.nn.bias_add(x1, self.G_b1)
    x1 = tf.reshape(x1, (-1,4,4,512))
    batch_mean, batch_var = tf.nn.moments(x1, [0])
    x1 = tf.nn.batch_normalization(x1, batch_mean, batch_var, self.bn1_beta, self.bn1_scale, epsilon)
    x1 = tf.maximum(alpha * x1, x1)
    shape = [batch_size,1,1,1]
    shape[1] = 8
    shape[2] = 8
    shape[3] = self.dc1.get_shape().as_list()[2]
    x2 = tf.nn.conv2d_transpose(x1, self.dc1, tf.stack(shape), strides=[1,2,2,1], padding="SAME")
    x2 = tf.nn.bias_add(x2, self.dc1_b)
    batch_mean, batch_var = tf.nn.moments(x2, [0])
    x2 = tf.nn.batch_normalization(x2, batch_mean, batch_var, self.bn2_beta, self.bn2_scale, epsilon)
    x2 = tf.maximum(alpha * x2, x2)
 
    shape = [batch_size,1,1,1]
    shape[1] = 16
    shape[2] = 16
    shape[3] = self.dc2.get_shape().as_list()[2]
    x3 = tf.nn.conv2d_transpose(x2, self.dc2, tf.stack(shape), strides=[1,2,2,1], padding="SAME")
    x3 = tf.nn.bias_add(x3, self.dc2_b)
    batch_mean, batch_var = tf.nn.moments(x3, [0])
    x3 = tf.nn.batch_normalization(x3, batch_mean, batch_var, self.bn3_beta, self.bn3_scale, epsilon)
    x3 = tf.maximum(alpha * x3, x3)
    
    shape = [batch_size,1,1,1]
    shape[1] = 50
    shape[2] = 50
    shape[3] = self.dc5.get_shape().as_list()[2]
    logits = tf.nn.conv2d_transpose(x3, self.dc5, tf.stack(shape), strides=[1,3,3,1], padding="VALID")
    output = tf.sigmoid(logits)
    output = tf.reshape(output,[-1,2500])
    return {"output": output }
   
  def get_variables(self):
    return [self.G_W1,self.G_b1,self.bn1_beta,self.bn1_scale,self.bn2_beta,self.bn2_scale,self.bn3_beta,self.bn3_scale,self.dc1,self.dc1_b,self.dc2,self.dc2_b,self.dc5,self.dc5_b]

class DCGAN_Dis(BaseModel):
  def create_model(self, input_size, **unused_params):
    #conv layer
    self.c1 = tf.Variable(tf.truncated_normal([5,5,1,64],stddev=0.1),name='d/bn1')
    self.b1 = tf.Variable(tf.zeros([64]),name='d/bn2')

    self.f = tf.Variable(tf.truncated_normal([25*25*64, 1],stddev=0.1),name='d/fn1')
    self.fb = tf.Variable(tf.zeros([1]),name='d/fn2')

  def run_model(self, model_input, is_training=True, **unused_params):
    alpha=0.2
    model_input = tf.reshape(model_input,[tf.shape(model_input)[0],50,50,1])
    x1 = tf.nn.conv2d(model_input, self.c1, strides=[1,2,2,1],padding="SAME")
    x1 = tf.nn.bias_add(x1,self.b1)
    x1 = tf.maximum(alpha*x1,x1)
    print(x1.shape) 
    x3 = tf.reshape(x1, [-1, 25*25*64])
    logits = tf.matmul(x3,self.f)
    logits = tf.nn.bias_add(logits,self.fb)
    predictions = tf.nn.sigmoid(logits)
    print(predictions.shape,"disoutshape")

    return {"logits":logits,"predictions":predictions}

  def get_variables(self):

    return [self.c1,self.b1,self.f,self.fb]

class DCGAN_Gend(BaseModel):
  def create_model(self, input_size, **unused_params):
    self.noise_input_size = 100    
    # linear layer
    h1_size = 4*4*512
    self.G_W1 = tf.Variable(xavier_init([self.noise_input_size, h1_size]), name='g/w1')
    self.G_b1 = tf.Variable(tf.zeros(shape=[h1_size]), name='g/b1')
    
    #batch_norm layer
    self.bn1_beta = tf.Variable(tf.zeros([512]),name='g/bn1')
    self.bn1_scale = tf.Variable(tf.ones([512]),name='g/bn1_s')
    self.bn2_beta = tf.Variable(tf.zeros([256]),name='g/bn2')
    self.bn2_scale = tf.Variable(tf.ones([256]),name='g/bn2_s')
    self.bn3_beta = tf.Variable(tf.zeros([128]),name='g/bn3')
    self.bn3_scale = tf.Variable(tf.ones([128]),name='g/bn3_s')    
    self.bn4_beta = tf.Variable(tf.zeros([128]),name='g/bn4')
    self.bn4_scale = tf.Variable(tf.ones([128]),name='g/bn4_s')


    #conv layer
    self.dc1 = tf.Variable(tf.random_normal([5,5,256,512],stddev=0.01),name='g/dc1')
    self.dc1_b = tf.Variable(tf.zeros([256]),name='g/dc1_b')
    self.dc2 = tf.Variable(tf.random_normal([5,5,128,256],stddev=0.01),name='g/dc2') 
    self.dc2_b = tf.Variable(tf.zeros([128]),name='g/dc2_b')
    self.dc3 = tf.Variable(tf.random_normal([5,5,128,128],stddev=0.01),name='g/dc3')
    self.dc3_b = tf.Variable(tf.zeros([128]),name='g/dn3_b')
    self.dc4 = tf.Variable(tf.random_normal([5,5,1,128],stddev=0.01),name='g/dc4')
    self.dc4_b = tf.Variable(tf.zeros([1]),name='g/dn4_b')

  def run_model(self, model_input, is_training=True, **unused_params):
    print(model_input.shape, "genshape")
    batch_size = tf.shape(model_input)[0]
    alpha = 0.2
    epsilon = 1e-3
    x1 = tf.matmul(model_input, self.G_W1)
    x1 = tf.nn.bias_add(x1, self.G_b1)
    # 4 x 4 x 512
    x1 = tf.reshape(x1, (-1,4,4,512))
    batch_mean, batch_var = tf.nn.moments(x1, [0])
    x1 = tf.nn.batch_normalization(x1, batch_mean, batch_var, self.bn1_beta, self.bn1_scale, epsilon)
    x1 = tf.maximum(alpha * x1, x1)

    # 8 x 8 x 256    
    shape = [batch_size,1,1,1]
    shape[1] = 8
    shape[2] = 8
    shape[3] = self.dc1.get_shape().as_list()[2]
    x2 = tf.nn.conv2d_transpose(x1, self.dc1, tf.stack(shape), strides=[1,2,2,1], padding="SAME")
    x2 = tf.nn.bias_add(x2, self.dc1_b)
    batch_mean, batch_var = tf.nn.moments(x2, [0])
    x2 = tf.nn.batch_normalization(x2, batch_mean, batch_var, self.bn2_beta, self.bn2_scale, epsilon)
    x2 = tf.maximum(alpha * x2, x2)
    # 16 x 16 x 128
    shape = [batch_size,1,1,1]
    shape[1] = 16
    shape[2] = 16
    shape[3] = self.dc2.get_shape().as_list()[2]
    x3 = tf.nn.conv2d_transpose(x2, self.dc2, tf.stack(shape), strides=[1,2,2,1], padding="SAME")
    x3 = tf.nn.bias_add(x3, self.dc2_b)
    batch_mean, batch_var = tf.nn.moments(x3, [0])
    x3 = tf.nn.batch_normalization(x3, batch_mean, batch_var, self.bn3_beta, self.bn3_scale, epsilon)
    x3 = tf.maximum(alpha * x3, x3)
    # 50 x 50 x 128 
    shape = [batch_size,1,1,1]
    shape[1] = 50
    shape[2] = 50
    shape[3] = self.dc3.get_shape().as_list()[2]
    logits = tf.nn.conv2d_transpose(x3, self.dc3, tf.stack(shape), strides=[1,3,3,1], padding="VALID")
    logits = tf.nn.bias_add(logits, self.dc3_b)
    batch_mean, batch_var = tf.nn.moments(logits, [0])
    logits = tf.nn.batch_normalization(logits, batch_mean, batch_var, self.bn4_beta, self.bn4_scale, epsilon)
    logits = tf.maximum(alpha * logits, logits)

    # 50 x 50 x 1
    shape = [batch_size,1,1,1]
    shape[1] = 50
    shape[2] = 50
    shape[3] = self.dc4.get_shape().as_list()[2]
    logits = tf.nn.conv2d_transpose(logits, self.dc4, tf.stack(shape), strides=[1,1,1,1], padding="SAME")
    logits = tf.nn.bias_add(logits, self.dc4_b)


    output = tf.sigmoid(logits)
    output = tf.reshape(output,[-1,2500])
    return {"output": output }
   
  def get_variables(self):
    return [self.G_W1,self.G_b1,self.bn1_beta,self.bn1_scale,self.bn2_beta,self.bn2_scale,self.bn3_beta,self.bn3_scale,self.bn4_beta,self.bn4_scale,self.dc1,self.dc1_b,self.dc2,self.dc2_b,self.dc3,self.dc3_b,self.dc4,self.dc4_b]

class DCGAN_Disd(BaseModel):
  def create_model(self, input_size, **unused_params):
    #conv layer
    self.c1 = tf.Variable(tf.truncated_normal([5,5,1,64],stddev=0.1),name='d/bn1')
    self.b1 = tf.Variable(tf.zeros([64]),name='d/bn2')

    self.f = tf.Variable(tf.truncated_normal([25*25*64, 1],stddev=0.1),name='d/fn1')
    self.fb = tf.Variable(tf.zeros([1]),name='d/fn2')

  def run_model(self, model_input, is_training=True, **unused_params):
    alpha=0.2
    model_input = tf.reshape(model_input,[tf.shape(model_input)[0],50,50,1])
    x1 = tf.nn.conv2d(model_input, self.c1, strides=[1,2,2,1],padding="SAME")
    x1 = tf.nn.bias_add(x1,self.b1)
    x1 = tf.maximum(alpha*x1,x1)
    print(x1.shape) 
    x3 = tf.reshape(x1, [-1, 25*25*64])
    logits = tf.matmul(x3,self.f)
    logits = tf.nn.bias_add(logits,self.fb)
    predictions = tf.nn.sigmoid(logits)
    print(predictions.shape,"disoutshape")

    return {"logits":logits,"predictions":predictions}

  def get_variables(self):

    return [self.c1,self.b1,self.f,self.fb]
`
