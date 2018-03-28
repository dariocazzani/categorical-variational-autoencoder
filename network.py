import tensorflow as tf
Bernoulli = tf.contrib.distributions.Bernoulli


def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y

class Network(object):

    # Create model
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.K = 10
        self.N = 30
        # temperature
        self.tau = tf.constant(5.0, name="temperature")
        self.lr = tf.constant(0.001, name="learning_rate")

        with tf.variable_scope("Encoder") as scope:
            # variational posterior q(y|x), i.e. the encoder (shape=(batch_size,200))
            self.logits_y = self.encoder(self.x, False)
            self.q_y = tf.nn.softmax(self.logits_y)
            self.log_q_y = tf.log(self.q_y + 1e-20)

        with tf.variable_scope("Decoder") as scope:
            self.y = tf.reshape(gumbel_softmax(self.logits_y, self.tau, hard=False),[-1, self.N, self.K])
            self.logits_x = self.decoder(self.y, False)
            self.p_x = Bernoulli(logits=self.logits_x)

        # Create loss
        self.loss = self.compute_loss()

    def encoder(self, x, reuse):
        inputs = tf.layers.dense(x, 512, reuse=reuse, activation=tf.nn.relu, name='fc1')
        inputs = tf.layers.dense(inputs, 256, reuse=reuse, activation=tf.nn.relu, name='fc2')
        inputs = tf.layers.dense(inputs, self.K * self.N, reuse=reuse, activation=None, name='logits_y')
        inputs = tf.reshape(inputs, [-1, self.K])
        return inputs

    def decoder(self, x, reuse):
        inputs = tf.layers.flatten(x)
        inputs = tf.layers.dense(inputs, 256, reuse=reuse, activation=tf.nn.relu, name='fc1')
        inputs = tf.layers.dense(inputs, 512, reuse=reuse, activation=tf.nn.relu, name='fc2')
        inputs = tf.layers.dense(inputs, 784, reuse=reuse, activation=None, name='fc3')
        return inputs

    def compute_loss(self):
        kl_tmp = tf.reshape(self.q_y * (self.log_q_y - tf.log(1.0/self.K)),[-1,self.N, self.K])
        KL = tf.reduce_sum(kl_tmp, [1,2])
        elbo=tf.reduce_sum(self.p_x.log_prob(self.x) ,1) - KL
        loss=tf.reduce_mean(-elbo)
        return loss
