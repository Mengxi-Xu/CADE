"""
autoencoder.py
~~~~~~~

Functions for training a unified autoencoder or individual autoencoders for each family.

"""

import os
os.environ['PYTHONHASHSEED'] = '0'
from numpy.random import seed
import random
random.seed(1)
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# TensorFlow wizardry
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5

import sys
import math
import time
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import logging

import cade.data as data
import cade.utils as utils
import cade.logger as logger


class Autoencoder(object):
    def __init__(self,
                 dims,
                 activation='relu',
                 init='glorot_uniform',
                 verbose=1):
        '''
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
        The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        activation: activation, not applied to Input, last layer of the encoder, and Output layers

        '''
        self.dims = dims
        self.act = activation
        self.init = init
        self.verbose = verbose

    def build(self):
        """Fully connected auto-encoder model, symmetric.

        return:
            (ae_model, encoder_model), Model of autoencoder and model of encoder
        """
        #这是一个使用全连接自编码器构建模型的函数。自编码器有两个部分，编码器和解码器，它们都由多个全连接层组成。
        #编码器将输入数据映射到隐藏表示，解码器将隐藏表示解码为原始输入数据。在该函数中，使用给定的维度、
        #激活函数和初始化方法定义了多个全连接层，构建了一个对称的自编码器模型。
        #模型包含一个输入层、一个输出层和多个隐藏层。同时，该函数还返回了自编码器模型和编码器模型，以便后续使用。
        dims = self.dims
        act = self.act
        init = self.init

        n_stacks = len(dims) - 1
        # input
        input_img = Input(shape=(dims[0],), name='input')
        x = input_img
        # internal layers in encoder
        #对于每个隐藏层，使用for循环添加编码器的内部层。这里使用的是全连接层Dense
        for i in range(n_stacks-1):
            #定义全连接层并添加到模型中。该层的节点数由dims[i+1]指定，激活函数由act指定，权重初始化方法由init指定。函数返回一个新的Keras层对象，并将其添加到模型中
            x = Dense(dims[i + 1], activation=act,
                      kernel_initializer=init, name='encoder_%d' % i)(x)
            # kernel_initializer is a fancy term for which statistical distribution or function to use for initializing the weights. Neural network needs to start with some weights and then iteratively update them

        # hidden layer, features are extracted from here, no activation is applied here, i.e., "linear" activation: a(x) = x
        #定义自编码器的编码器部分的最后一层，并将其赋给变量encoded。这里没有激活函数，因为我们只需要提取特征并减少数据维度
        encoded = Dense(dims[-1], kernel_initializer=init,
                        name='encoder_%d' % (n_stacks - 1))(x)
        #将编码器输出存储到实例变量self.encoded中，以便后续使用。
        self.encoded = encoded

        x = encoded
        # internal layers in decoder
        #从最后一层解码器往前逐层解码
        for i in range(n_stacks-1, 0, -1):
            x = Dense(dims[i], activation=act,
                      kernel_initializer=init, name='decoder_%d' % i)(x)

        # output
        x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
        decoded = x
        self.out = decoded

        ae = Model(inputs=input_img, outputs=decoded, name='AE')
        encoder = Model(inputs=input_img, outputs=encoded, name='encoder')
        return ae, encoder

    def train_and_save(self, X,
                       weights_save_name,
                       lr=0.001,
                       batch_size=32,
                       epochs=250,
                       loss='mse'):
        if os.path.exists(weights_save_name):#判断是否已存在保存权重的文件，如果存在则无需训练纯自编码器。
            logging.info('weights file exists, no need to train pure AE')
        else:
            logging.debug(f'AE train_and_save lr: {lr}')
            logging.debug(f'AE train_and_save batch_size: {batch_size}')
            logging.debug(f'AE train_and_save epochs: {epochs}')

            verbose = self.verbose
            #构建自编码器和编码器模型
            autoencoder, encoder = self.build()
            #构建自编码器和编码器模型
            pretrain_optimizer = Adam(lr=lr)
            #编译自编码器模型，指定优化器和损失函数。
            autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
            #创建保存权重的文件所在目录。
            utils.create_parent_folder(weights_save_name)
            #定义训练过程中的回调函数，用于保存最优模型
            mcp_save = ModelCheckpoint(weights_save_name,
                                    monitor='loss',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    verbose=verbose,
                                    mode='min')

            hist = autoencoder.fit(X, X,
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=1,
                                callbacks=[mcp_save, logger.LoggingCallback(logging.debug)])

    def evaluate_quality(self, X_old, y_old, model_save_name):
        if not os.path.exists(model_save_name):
            self.train_and_save(X_old, model_save_name)

        K.clear_session()
        autoencoder, encoder = self.build()
        encoder.load_weights(model_save_name, by_name=True)
        logging.debug(f'Load weights from {model_save_name}')
        latent = encoder.predict(X_old)

        best_acc = 0
        best_n_init = 10
        num_classes = len(np.unique(y_old))
        logging.debug(f'KMeans k = {num_classes}')

        warnings.filterwarnings('ignore')

        for n_init in range(10, 110, 10):
            kmeans = KMeans(n_clusters=num_classes, n_init=n_init,
                            random_state=42, n_jobs=-1)
            y_pred = kmeans.fit_predict(latent)
            acc = utils.get_cluster_acc(y_old, y_pred)
            logging.debug(f'KMeans n_init: {n_init}, acc: {acc}')
            if acc > best_acc:
                best_n_init = best_n_init
                best_acc = acc
        logging.info(f'best accuracy of KMeans on latent data: {best_acc} with n_init {best_n_init}')
        return best_acc


class ContrastiveAE(object):
    def __init__(self, dims, optimizer, lr, verbose=1):
        self.dims = dims
        self.optimizer = optimizer(lr)
        self.verbose = verbose

    def train(self, X_train, y_train,
              lambda_1, batch_size, epochs, similar_ratio, margin,
              weights_save_name, display_interval):
        """Train an autoencoder with standard mse loss + contrastive loss.

        Arguments:
            X_train {numpy.ndarray} -- feature vectors of the training data
            y_train {numpy.ndarray} -- ground-truth labels of the training data
            lambda_1 {float} -- balance factor for the autoencoder reconstruction loss and contrastive loss
            batch_size {int} -- number of samples in each batch (note we only use **half of batch_size**
                                from the training data).
            epochs {int} -- No. of maximum epochs.
            similar_ratio {float} -- ratio of similar samples, use 0.25 for now.
            margin {float} -- the hyper-parameter m.
            weights_save_name {str} -- file path to save the best weights files.
            display_interval {int} -- print traning logs per {display_interval} epoches
        """
        if os.path.exists(weights_save_name):
            logging.info('weights file exists, no need to train contrastive AE')
        else:
            tf.reset_default_graph()

            labels = tf.placeholder(tf.float32, [None])
            lambda_1_tensor = tf.placeholder(tf.float32)
            ae = Autoencoder(self.dims)
            ae_model, encoder_model = ae.build()

            input_ = ae_model.get_input_at(0)

            # add loss function -- for efficiency and not doubling the network's weights, we pass a batch of samples and
            # make the pairs from it at the loss level.
            left_p = tf.convert_to_tensor(list(range(0, int(batch_size / 2))), np.int32)
            right_p = tf.convert_to_tensor(list(range(int(batch_size / 2), batch_size)), np.int32)

            # left_p: indices with all the data in this batch, right_p: half with similar data compared to left_p, half with dissimilar data compared to left_p
            # if batch_size = 16 (but only using 8 samples in this batch):
            # e.g., left_p labels: 1, 2, 4, 8 | 2, 3, 5, 6
            #      right_p labels: 1, 2, 4, 8 | 3, 4, 1, 7
            # check whether labels[left_p] == labels[right_p] for each element
            is_same = tf.cast(tf.equal(tf.gather(labels, left_p), tf.gather(labels, right_p)), tf.float32)
            # NOTE: add a small number like 1e-10 would prevent tf.sqrt() to have 0 values, further leading gradients and loss all NaN.
            # check: https://stackoverflow.com/questions/33712178/tensorflow-nan-bug
            dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf.gather(ae.encoded, left_p), tf.gather(ae.encoded, right_p))), 1) + 1e-10) # ||zi - zj||_2
            contrastive_loss = tf.multiply(is_same, dist) # y_ij = 1 means the same class.
            contrastive_loss = contrastive_loss + tf.multiply((tf.constant(1.0) - is_same), tf.nn.relu(margin - dist))  # as relu(z) = max(0, z)
            contrastive_loss = tf.reduce_mean(contrastive_loss)

            ae_loss = tf.keras.losses.MSE(input_, ae.out) # ae.out equals ae_model(input_)
            ae_loss = tf.reduce_mean(ae_loss)

            # Final loss
            loss = lambda_1 * contrastive_loss + ae_loss

            train_op = self.optimizer.minimize(loss, var_list=tf.trainable_variables())

            # Start training
            with tf.Session(config=config) as sess:# 创建一个新的TensorFlow会话
                loss_batch, aux_batch = [], []# 定义损失和辅助列表，用于记录训练过程中的值
                contrastive_loss_batch, ae_loss_batch = [], []
                sess.run(tf.global_variables_initializer()) # 初始化可训练变量

min_loss = np.inf # 初始化最小损失为正无穷大

# epoch training loop 进行epochs个周期的训练
for epoch in range(epochs):
    epoch_time = time.time()
    
    # split data into batches 将训练数据分为batch_count个批次
    batch_count, batch_x, batch_y = data.epoch_batches(X_train, y_train,
                                                           batch_size,
                                                           similar_ratio)
    # batch training loop 在每个批次中执行训练操作
    for b in range(batch_count):
        logging.debug(f'b: {b}')
        feed_dict = {
            input_: batch_x[b],
            labels: batch_y[b],
            lambda_1_tensor: lambda_1
        }
        # 计算损失、训练操作和其他值
        loss1, _, aux1, contrastive_loss1, ae_loss1, \
            dist1, encoded1 = sess.run([loss, train_op, is_same, contrastive_loss, ae_loss,
                                        dist, ae.encoded], feed_dict=feed_dict)

        logging.debug(f'loss1: {loss1},  aux1: {aux1}')
        logging.debug(f'contrastive: {contrastive_loss1}, ae: {ae_loss1}')
        logging.debug(f'epoch-{epoch} dist1[left]: {dist1[0:batch_size // 4]}')
        logging.debug(f'epoch-{epoch} dist1[right]: {dist1[batch_size // 4:]}')

        # 记录损失和其他值到对应的列表中
        loss_batch.append(loss1)
        aux_batch.append(aux1)
        contrastive_loss_batch.append(contrastive_loss1)
        ae_loss_batch.append(ae_loss1)

    if math.isnan(np.mean(loss_batch)):
        logging.error('NaN value in loss')

    # print logs each xxx epoch 每xxx个周期打印日志
    if epoch % display_interval == 0:
        current_loss = np.mean(loss_batch) # 计算当前损失值的平均值
        # 打印训练日志并记录损失、辅助值和各种损失的平均值
        logging.info(f'Epoch {epoch}: loss {current_loss} -- ' + \
                    f'contrastive {np.mean(contrastive_loss_batch)} -- ' + \
                    f'ae {np.mean(ae_loss_batch)} -- ' + \
                    f'pairs {np.mean(np.sum(np.mean(aux_batch)))} : ' + \
                    f'{np.mean(np.sum(1-np.mean(aux_batch)))} -- ' + \
                    f'time {time.time() - epoch_time}')
        loss_batch, aux_batch = [], [] # 清空损失和辅助列表以开始下一个周期的训练
        contrastive_loss_batch, ae_loss_batch = [], []

        # save best weights 如果当前损失小于最小损失，则更新最佳权重并保存它们
        if current_loss < min_loss:
            logging.info(f'updating best loss from {min_loss} to {current_loss}')
            min_loss = current_loss
            ae_model.save_weights(weights_save_name)
