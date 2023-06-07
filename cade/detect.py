"""
detect.py
~~~~~~~

Functions for detecting drifting samples, write the closest family for each sample in the testing set.

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
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from collections import Counter
from keras import backend as K
from keras.models import load_model
from tqdm import tqdm

from sklearn.manifold import TSNE

import cade.utils as utils
from cade.autoencoder import Autoencoder


def detect_drift_samples(X_train, y_train, X_test, y_test, y_pred,
                       dims,
                       margin,
                       mad_threshold,
                       best_weights_file,
                       all_detect_path, simple_detect_path,
                       training_info_for_detect_path):
    # 如果检测结果文件已经存在，则无需重新进行检测
    if os.path.exists(all_detect_path) and os.path.exists(simple_detect_path):
        logging.info('Detection result files exist, no need to redo the detection')
    else:
        '''get latent data for the entire training and testing set'''
        # 获取训练集和测试集在latent空间中的表示
        z_train, z_test = get_latent_representation_keras(dims, best_weights_file, X_train, X_test)

        '''get latent data for each family in the training set'''
        # 将训练集分为N个家族，每个家族都有Ni个样本，并计算每个家族的中心点
        N, N_family, z_family = get_latent_data_for_each_family(z_train, y_train)

        '''get centroid for each family in the latent space'''
        # 获取每个家族在latent空间中的中心点
        centroids = [np.mean(z_family[i], axis=0) for i in range(N)]
        logging.debug(f'centroids: {centroids}')

        '''get distance between each training sample and their family's centroid in the latent space '''
        # 计算每个训练样本与其所属家族中心点之间的距离
        dis_family = get_latent_distance_between_sample_and_centroid(z_family, centroids,
                                                                     margin,
                                                                     N, N_family)

        '''get the MAD for each family'''
        # 计算每个家族的异常得分
        mad_family = get_MAD_for_each_family(dis_family, N, N_family)

        # 保存一些用于检测的训练信息，如训练集在latent空间中的表示，每个家族的中心点和异常得分等
        np.savez_compressed(training_info_for_detect_path,
                            z_train=z_train,
                            z_family=z_family,
                            centroids=centroids,
                            dis_family=dis_family,
                            mad_family=mad_family)

        '''detect drifting in the testing set'''
        # 对测试集中的每个样本进行检测，并将结果写入文件中
        with open(all_detect_path, 'w') as f1:
            f1.write('sample_idx,is_drift,closest_family,real_label,pred_label,min_distance,min_anomaly_score\n')
            with open(simple_detect_path, 'w') as f2:
                f2.write('sample_idx,closest_family,real_label,pred_label,min_distance,min_anomaly_score\n')

                for k in tqdm(range(len(X_test)), desc='detect', total=X_test.shape[0]):
                    z_k = z_test[k]
                    '''get distance between each testing sample and each centroid'''
                    # 计算测试样本与每个家族中心点之间的距离
                    dis_k = [np.linalg.norm(z_k - centroids[i]) for i in range(N)]
                    # 使用距离计算每个家族的异常得分
                    anomaly_k = [np.abs(dis_k[i] - np.median(dis_family[i])) / mad_family[i] for i in range(N)]
                    logging.debug(f'sample-{k} - dis_k: {dis_k}')
                    logging.debug(f'sample-{k} - anomaly_k: {anomaly_k}')

                    # 根据异常得分判断是否发生了漂移，将结果写入文件中
                    #np.argmin(dis_k) 返回数组 dis_k 中最小元素的索引。
                    #np.min(dis_k) 返回数组 dis_k 中的最小值。
                    #np.min(anomaly_k) 返回数组 anomaly_k 中的最小值。
                    closest_family = np.argmin(dis_k)
                    min_dis = np.min(dis_k)
                    min_anomaly_score = np.min(anomaly_k)

                    if min_anomaly_score > mad_threshold:
                        logging.debug(f'testing sample {k} is drifting')
                        f1.write(f'{k},Y,{closest_family},{y_test[k]},{y_pred[k]},{min_dis},{min_anomaly_score}\n')
                        f2.write(f'{k},{closest_family},{y_test[k]},{y_pred[k]},{min_dis},{min_anomaly_score}\n')
                    else:
                        f1.write(f'{k},N,{closest_family},{y_test[k]},{y_pred[k]},{min_dis},{min_anomaly_score}\n')

def get_latent_representation_keras(dims, best_weights_file, X_train, X_test):
    K.clear_session()
    ae = Autoencoder(dims)
    ae_model, encoder_model = ae.build()
    encoder_model.load_weights(best_weights_file, by_name=True)

    z_train = encoder_model.predict(X_train)
    z_test = encoder_model.predict(X_test)

    logging.debug(f'z_train shape: {z_train.shape}')
    logging.debug(f'z_test shape: {z_test.shape}')
    logging.debug(f'z_train[0]: {z_train[0]}')

    return z_train, z_test


def get_latent_data_for_each_family(z_train, y_train):
    N = len(np.unique(y_train))
    N_family = [len(np.where(y_train == family)[0]) for family in range(N)]
    z_family = []
    for family in range(N):
        z_tmp = z_train[np.where(y_train == family)[0]]
        z_family.append(z_tmp)

    z_len = [len(z_family[i]) for i in range(N)]
    logging.debug(f'z_family length: {z_len}')

    return N, N_family, z_family


def get_latent_distance_between_sample_and_centroid(z_family, centroids, margin, N, N_family):
    dis_family = []  # two-dimension list

    for i in range(N): # i: family index
        dis = [np.linalg.norm(z_family[i][j] - centroids[i]) for j in range(N_family[i])]
        dis_family.append(dis)

    dis_len = [len(dis_family[i]) for i in range(N)]
    logging.debug(f'dis_family length: {dis_len}')

    return dis_family


def get_MAD_for_each_family(dis_family, N, N_family):
    mad_family = []
    for i in range(N):
        median = np.median(dis_family[i])
        logging.debug(f'family {i} median: {median}')
        diff_list = [np.abs(dis_family[i][j] - median) for j in range(N_family[i])]
        mad = 1.4826 * np.median(diff_list)  # 1.4826: assuming the underlying distribution is Gaussian
        mad_family.append(mad)
    logging.debug(f'mad_family: {mad_family}')

    return mad_family
