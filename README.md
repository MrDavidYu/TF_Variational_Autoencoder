# TF_Variational_Autoencoder
#### _Variational autoencoder for compressing/reconstructing RGB inputs (CelebA) in TensorFlow with high compression_

N.B. This project is mostly a clone of my previous project https://github.com/MrDavidYu/TF_Convolutional_Autoencoder and so will be very sparce in its documentation. Please reference the link for more detail.

The following changes are made to this project on top of a Convolutional Autoencoder:
1. The following mean and stddev layers are added to the bottleneck layer:
```
fc_mu = fullyConnected(do3, name='fc_mu', output_size=latent_dim)   # Input: [12*12*3];  Output: [64] --> mean_vec
fc_log_sig = fullyConnected(do3, name='fc_sig', output_size=latent_dim) # Input: [12*12*3];  Output: [64] --> stddev_vec
fc3 = fullyConnectedSample(fc_mu, fc_log_sig, name='fc3', output_size=latent_dim)  # Sample layer, Input/Output: [64]
```
Where the fullyConnectSample() function returns a sample using the random variable X with mean and stddev specified by the layers fc_mu and fc_log_sig respectively.

2. The MSE loss function has been replaced with vae_loss with the following definition:
```
def vae_loss(input_img, output, input_mu, input_log_sig):
  reconstruction_loss = tf.reduce_sum(tf.square(tf.subtract(output, tf.reshape(x,shape=[-1,48*48*3]))), axis=1)
  kl_loss = -0.5 * tf.reduce_sum(1+input_log_sig-tf.square(input_mu)-tf.exp(input_log_sig), axis=1)
  total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
  return total_loss
```
3. A block of (commented) code for visualizing the bottleneck layer has been added in the training step. This block of code requires the batch size to be 111 and const_latent_dim to be 2, since we can only view outputs in 2D. This limitation also means that the model can potentially become unstable in training and lead to an exploding gradient problem. This was encountered after 2000 steps of training, but not before some good visual clusters were produced by the bottleneck layer (see following illustration). Note in application the const_latent_dim should be kept at 64 (or larger) for optimal convergence.

## Clustering Visualization
The following graphs are the difference in RGB channels (in the latent tensor) between step 1 and step 2000. Note for regression RGB input it is difficult to produce clear clusters as one would expect from say MNIST, hence the following serves as only a crude demonstration of the ability of VAEs to roughly produce coherent clusters based on value averages of each color channel.
<img src="https://github.com/MrDavidYu/TF_Variational_Autoencoder/blob/master/sample_output/scatter_1.png" height="320" />
<img src="https://github.com/MrDavidYu/TF_Variational_Autoencoder/blob/master/sample_output/scatter_2000.png" height="320" />
