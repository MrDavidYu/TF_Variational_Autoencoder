"""
TF Convolutional Autoencoder

Arash Saber Tehrani - May 2017
Reference: https://github.com/arashsaber/Deep-Convolutional-AutoEncoder

Modified David Yu - July 2018
Reference: https://github.com/MrDavidYu/TF_Convolutional_Autoencoder
Add ons:
1. Allows for custom .jpg input
2. Checkpoint save/restore
3. TensorBoard logs for input/output images
3. Input autorescaling
4. ReLU activation replaced by LeakyReLU

"""
import os
import re
import scipy.misc
import numpy as np
import matplotlib
# N.B. This is used to force matplotlib to not use any Xwindows backend. Needed for training on remote machine.
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob


# Some important consts
const_latent_dim = 64  # Let this be 64 for training, and 2 for visualizing the latent space in 2D.
num_examples = 669
batch_size = 60
n_epochs = 1000
save_steps = 500  # Number of training batches between checkpoint saves

checkpoint_dir = "./ckptv/"
model_name = "ConvAutoEnc.model"
logs_dir = "./logsv/run1/"

# Fetch input data (faces/trees/imgs)
data_dir = "./data/celebG/"
data_path = os.path.join(data_dir, '*.jpg')
data = glob(data_path)

if len(data) == 0:
    raise Exception("[!] No data found in '" + data_path+ "'")


'''
Some util functions from https://github.com/carpedm20/DCGAN-tensorflow
'''

def path_to_img(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
              resize_height=48, resize_width=48, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def autoresize(image_path, input_height, input_width,
              resize_height=48, resize_width=48,
              crop=True, grayscale=False):
  image = path_to_img(image_path, grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

np.random.shuffle(data)
imread_img = path_to_img(data[0])  # test read an image

if len(imread_img.shape) >= 3: # check if image is a non-grayscale image by checking channel number
    c_dim = path_to_img(data[0]).shape[-1]
else:
    c_dim = 1

is_grayscale = (c_dim == 1)

'''
tf Graph Input
'''
x = tf.placeholder(tf.float32, [None, 48, 48, 3], name='InputData')

if __debug__:
    print("Reading input from:" + data_dir)
    print("Input image shape:" + str(imread_img.shape))
    print("Assigning input tensor of shape:" + str(x.shape))
    print("Writing checkpoints to:" + checkpoint_dir)
    print("Writing TensorBoard logs to:" + logs_dir)


# strides = [Batch, Height, Width, Channels]  in default NHWC data_format. Batch and Channels
# must always be set to 1. If channels is set to 3, then we would increment the index for the
# color channel by 3 everytime we convolve the filter. So this means we would only use one of
# the channels and skip the other two. If we change the Batch number then it means some images
# in the batch are skipped.
#
# To calculate the size of the output of CONV layer:
# OutWidth = (InWidth - FilterWidth + 2*Padding)/Stride + 1
def conv2d(input, name, kshape, strides=[1, 1, 1, 1]):
    with tf.variable_scope(name):
        W = tf.get_variable(name='w_' + name,
                            shape=kshape,
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(name='b_' + name,
                            shape=[kshape[3]],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        out = tf.nn.conv2d(input,W,strides=strides, padding='SAME')
        out = tf.nn.bias_add(out, b)
        out = tf.nn.leaky_relu(out)
        return out


# tf.contrib.layers.conv2d_transpose, do not get confused with 
# tf.layers.conv2d_transpose
def deconv2d(input, name, kshape, n_outputs, strides=[1, 1]):
    with tf.variable_scope(name):
        out = tf.contrib.layers.conv2d_transpose(input,
                 num_outputs= n_outputs,
                 kernel_size=kshape,
                 stride=strides,
                 padding='SAME',
                 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                 biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                 activation_fn=tf.nn.leaky_relu)
        return out


# Input to maxpool: [BatchSize, Width1, Height1, Channels]
# Output of maxpool: [BatchSize, Width2, Height2, Channels]
#
# To calculate the size of the output of maxpool layer:
# OutWidth = (InWidth - FilterWidth)/Stride + 1
#
# The kernel kshape will typically be [1,2,2,1] for a general 
# RGB image input of [batch_size,48,48,3]
# kshape is 1 for batch and channels because we don't want to take
# the maximum over multiple examples of channels.
def maxpool2d(x,name,kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    with tf.variable_scope(name):
        out = tf.nn.max_pool(x,
                 ksize=kshape, #size of window
                 strides=strides,
                 padding='SAME')
        return out


def upsample(input, name, factor=[2,2]):
    size = [int(input.shape[1] * factor[0]), int(input.shape[2] * factor[1])]
    with tf.variable_scope(name):
        out = tf.image.resize_bilinear(input, size=size, align_corners=None, name=None)
        return out


def fullyConnected(input, name, output_size):
    with tf.variable_scope(name):
        input_size = input.shape[1:]
        input_size = int(np.prod(input_size)) # get total num of cells in one input image
        W = tf.get_variable(name='w_'+name,
                shape=[input_size, output_size],
                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(name='b_'+name,
                shape=[output_size],
                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        input = tf.reshape(input, [-1, input_size])
        out = tf.nn.leaky_relu(tf.add(tf.matmul(input, W), b))
        return out


def fullyConnectedSample(input_mu, input_log_sig, name, output_size):
    with tf.variable_scope(name):
        std_norm = tf.random_normal(shape=[batch_size, output_size], mean=0.0, stddev=1.0)
        return input_mu + tf.exp(input_log_sig) * std_norm


def dropout(input, name, keep_rate):
    with tf.variable_scope(name):
        out = tf.nn.dropout(input, keep_rate)
        return out

def vae_loss(input_img, output, input_mu, input_log_sig):
    # reconstruction_loss is simply the MSE from convolutional autoencoder
    # reconstruction_loss = K.sum(K.square(output-input_img))
    reconstruction_loss = tf.reduce_sum(tf.square(tf.subtract(output, tf.reshape(x,shape=[-1,48*48*3]))), axis=1)
    # compute the KL loss
    # kl_loss = - 0.5 * K.sum(1 + log_stddev - K.square(mean) - K.square(K.exp(log_stddev)), axis=-1)
    kl_loss = -0.5 * tf.reduce_sum(1+input_log_sig-tf.square(input_mu)-tf.exp(input_log_sig), axis=1)
    
    # return the average loss over all images in batch
    # total_loss = K.mean(reconstruction_loss + kl_loss)
    # N.B. kl_loss could be applied a weight of 0.005
    total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    return total_loss


def ConvAutoEncoder(x, name, latent_dim=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        input = tf.reshape(x, shape=[-1, 48, 48, 3])

        # kshape = [k_h, k_w, in_channels, out_chnnels]
        c1 = conv2d(input, name='c1', kshape=[7, 7, 3, 15])         # Input: [48,48,3];  Output: [48,48,15]
        p1 = maxpool2d(c1, name='p1')                               # Input: [48,48,15]; Output: [24,24,15]
        do1 = dropout(p1, name='do1', keep_rate=0.75)
        c2 = conv2d(do1, name='c2', kshape=[5, 5, 15, 25])          # Input: [24,24,15]; Output: [24,24,25]
        p2 = maxpool2d(c2, name='p2')                               # Input: [24,24,25]; Output: [12,12,25]
        p2 = tf.reshape(p2, shape=[-1, 12*12*25])                   # Input: [12,12,25]; Output: [12*12*25]
        fc1 = fullyConnected(p2, name='fc1', output_size=12*12*5)   # Input: [12*12*25]; Output: [12*12*5]
        do2 = dropout(fc1, name='do2', keep_rate=0.75)
        fc2 = fullyConnected(do2, name='fc2', output_size=12*12*3)  # Input: [12*12*5];  Output: [12*12*3]
        do3 = dropout(fc2, name='do3', keep_rate=0.75)
        fc_mu = fullyConnected(do3, name='fc_mu', output_size=latent_dim)   # Input: [12*12*3];  Output: [64] --> mean_vec
        fc_log_sig = fullyConnected(do3, name='fc_sig', output_size=latent_dim) # Input: [12*12*3];  Output: [64] --> stddev_vec
        fc3 = fullyConnectedSample(fc_mu, fc_log_sig, name='fc3', output_size=latent_dim)  # Sample layer, Input/Output: [64]
        # Decoding part
        fc4 = fullyConnected(fc3, name='fc4', output_size=12*12*3)  # Input: [64];       Output: [12*12*3]
        do4 = dropout(fc4, name='do4', keep_rate=0.75)
        fc5 = fullyConnected(do4, name='fc5', output_size=12*12*5)  # Input: [12*12*3];  Output: [12*12*5]
        do5 = dropout(fc5, name='do5', keep_rate=0.75)
        fc6 = fullyConnected(do5, name='fc6', output_size=21*21*25) # Input: [12*12*5];  Output: [12*12*25]
        do6 = dropout(fc6, name='do6', keep_rate=0.75)
        do6 = tf.reshape(do6, shape=[-1, 21, 21, 25])               # Input: [12*12*25]; Output: [12,12,25]
        dc1 = deconv2d(do6, name='dc1', kshape=[5, 5],n_outputs=15) # Input: [12,12,25]; Output: [12,12,15]
        up1 = upsample(dc1, name='up1', factor=[2, 2])              # Input: [12,12,15]; Output: [24,24,15]
        dc2 = deconv2d(up1, name='dc2', kshape=[7, 7],n_outputs=3)  # Input: [24,24,15]; Output: [24,24,3]
        up2 = upsample(dc2, name='up2', factor=[2, 2])              # Input: [24,24,3];  Output: [48,48,3]
        output = fullyConnected(up2, name='output', output_size=48*48*3)

        with tf.variable_scope('cost'):
            # N.B. reduce_mean is a batch operation! finds the mean across the batch
            cost = tf.reduce_mean(tf.square(tf.subtract(output, tf.reshape(x,shape=[-1,48*48*3]))))
        return x, tf.reshape(output,shape=[-1,48,48,3]), tf.reshape(fc3,shape=[-1,2]), cost # return input, output, latent space and cost


def train_network(x):

    with tf.Session() as sess:

        _, _, latent, cost = ConvAutoEncoder(x, 'ConvAutoEnc', latent_dim=const_latent_dim)
        with tf.variable_scope('opt'):
            optimizer = tf.train.AdamOptimizer().minimize(cost)

        # Create a summary to monitor cost tensor
        tf.summary.scalar("cost", cost)
        tf.summary.image("face_input", ConvAutoEncoder(x, 'ConvAutoEnc', reuse=True, latent_dim=const_latent_dim)[0], max_outputs=4)
        tf.summary.image("face_output", ConvAutoEncoder(x, 'ConvAutoEnc', reuse=True, latent_dim=const_latent_dim)[1], max_outputs=4)
        merged_summary_op = tf.summary.merge_all()  # Merge all summaries into a single op

        sess.run(tf.global_variables_initializer())  # memory allocation exceeded 10% issue

        # Model saver
        saver = tf.train.Saver()

        counter = 0  # Used for checkpointing
        success, restored_counter = restore(saver, sess)
        if success:
            counter = restored_counter
            print(">>> Restore successful")
        else:
            print(">>> No restore checkpoints detected")        

        # create log writer object
        writer = tf.summary.FileWriter(logs_dir, graph=tf.get_default_graph())

        for epoch in range(n_epochs):
            avg_cost = 0
            n_batches = int(num_examples / batch_size)
            # Loop over all batches
            for i in range(n_batches):
                counter += 1
                print("epoch " + str(epoch) + " batch " + str(i) + " counter " + str(counter))

                batch_files = data[i*batch_size:(i+1)*batch_size]  # get the current batch of files

                batch = [autoresize(batch_file,
                                        input_height=48,
                                        input_width=48,
                                        resize_height=48,
                                        resize_width=48,
                                        crop=True,
                                        grayscale=False) for batch_file in batch_files]

                batch_images = np.array(batch).astype(np.float32)

                # Get cost function from running optimizer
                _, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={x: batch_images})

                # Compute average loss
                avg_cost += c / n_batches

                writer.add_summary(summary, epoch * n_batches + i)

                if counter % save_steps == 0:
                    save(saver, counter, sess)

                ''' Visualize latent layer. Comment out this block in training '''
                # if counter == 1 or counter % 2000 == 0:
                #     z = tf.placeholder(tf.float32, [batch_size, 2])
                #     z = ConvAutoEncoder(batch_images, 'ConvAutoEnc', reuse=True, latent_dim=const_latent_dim)[2]
                #     zz = z.eval()
                #     # print("zz.shape = " + str(zz.shape))
                #     # plt.figure(figsize=(10, 8))
                #     f, axarr = plt.subplots(2, 2)
                #     # print(batch_images.shape)
                #     # print(batch_images[0].shape)
                #     # print(batch_images[1].shape)
                #     colors = np.random.rand(batch_size)
                #     R = np.zeros(shape=(batch_size))
                #     for j in range(batch_size):  # red space
                #         tmp_sum = 0
                #         for k in range(48):
                #             for l in range(48):
                #                 tmp_sum += batch_images[j][k][l][0]
                #         R[j] = tmp_sum / (48*48)
                #     G = np.zeros(shape=(batch_size))
                #     for j in range(batch_size):  # green space
                #         tmp_sum = 0
                #         for k in range(48):
                #             for l in range(48):
                #                 tmp_sum += batch_images[j][k][l][1]
                #         G[j] = tmp_sum / (48*48)
                #     B = np.zeros(shape=(batch_size))
                #     for j in range(batch_size):  # blue space
                #         tmp_sum = 0
                #         for k in range(48):
                #             for l in range(48):
                #                 tmp_sum += batch_images[j][k][l][2]
                #         B[j] = tmp_sum / (48*48)
                #     # plt.subplot(2,2,1)
                #     axarr[0, 0].scatter(zz[:, 0], zz[:, 1], c=R)
                #     axarr[0, 0].grid()
                #     axarr[0, 0].set_title("red")
                #     axarr[0, 1].scatter(zz[:, 0], zz[:, 1], c=G)
                #     axarr[0, 1].grid()
                #     axarr[0, 1].set_title("green")
                #     axarr[1, 0].scatter(zz[:, 0], zz[:, 1], c=B)
                #     axarr[1, 0].grid()
                #     axarr[1, 0].set_title("blue")
                #     plt.grid()
                #     plt.savefig("scatter_"+str(counter)+".png")
                ''' Visualize latent layer end '''

            # Display logs per epoch step
            print('Epoch', epoch + 1, ' / ', n_epochs, 'cost:', avg_cost)

        print('>>> Optimization Finished')


# Create checkpoint
def save(saver, step, session):
    print(">>> Saving to checkpoint, step:" + str(step))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(session,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)


# Restore from checkpoint
def restore(saver, session):
    print(">>> Restoring from checkpoints...")
    checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint_state and checkpoint_state.model_checkpoint_path:
      checkpoint_name = os.path.basename(checkpoint_state.model_checkpoint_path)
      saver.restore(session, os.path.join(checkpoint_dir, checkpoint_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",checkpoint_name)).group(0))
      print(">>> Found restore checkpoint {}".format(checkpoint_name))
      return True, counter
    else:
      return False, 0

train_network(x)
