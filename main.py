import os
os.environ['PYTHONHASHSEED'] = '0'
from numpy.random import seed
import random
random.seed(1)
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import sys
import logging
import traceback

from timeit import default_timer as timer
from pprint import pformat
from collections import Counter
from keras.models import load_model
from keras import backend as K
from keras.utils import np_utils
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cade.data as data
import cade.detect as detect
import cade.utils as utils
import cade.classifier as classifier
import cade.evaluate as evaluate
import cade.explain_by_distance as explain_dis
from cade.autoencoder import ContrastiveAE, Autoencoder
from cade.logger import init_log

import numpy as np
import tensorflow as tf

# TensorFlow wizardry
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))


def main():
    # ---------------------------------------- #
    # 0. Init log path and parse args          #
    # ---------------------------------------- #

    args = utils.parse_args()#//解析命令行参数的函数，返回一个包含命令行参数的对象

    log_path = './logs/main'
    if args.quiet:
        init_log(log_path, level=logging.INFO)
    else:
        init_log(log_path, level=logging.DEBUG)
    logging.warning('Running with configuration:\n' + pformat(vars(args)))
    logging.getLogger('matplotlib.font_manager').disabled = True

    logging.debug(f'available GPUs: {K.tensorflow_backend._get_available_gpus()}')

    # ----------------------------------------------- #
    # 1. Prepare the dataset                          #
    # ----------------------------------------------- #

    data.prepare_dataset(args)

    # ---------------------------------------- #
    # 2. Load the feature vectors and labels   #
    # ---------------------------------------- #

    logging.info(f'Loading {args.data} dataset')

    X_train, y_train, X_test, y_test = data.load_features(args.data, args.newfamily_label)

    logging.info(f'Loaded train: {X_train.shape}, {y_train.shape}')
    logging.info(f'Loaded test: {X_test.shape}, {y_test.shape}')
    logging.info(f'y_train labels: {np.unique(y_train)}')
    logging.info(f'y_test labels: {np.unique(y_test)}')
    logging.info(f'y_train: {Counter(y_train)}')
    logging.info(f'y_test: {Counter(y_test)}')
    #//使用日志记录模块输出一些有关数据集的基本信息，
    #//包括训练集和测试集的大小、标签的取值范围以及每个标签对应的样本数量（使用 Counter 函数统计）

    # ----------------------------------------------- #
    # 3. Train classifier and evaluate on test set    #
    # ----------------------------------------------- #

    # some commonly used variables.
    SAVED_MODEL_FOLDER = 'models/'
    DATA_FOLDER = 'data/'
    if args.pure_ae == 0:
        REPORT_FOLDER = 'reports/'
        FIG_FOLDER = 'fig/'
    else:
        REPORT_FOLDER = 'pure_ae_reports/'
        FIG_FOLDER = 'pure_ae_fig/'
    NUM_FEATURES = X_train.shape[1]
    NUM_CLASSES = len(np.unique(y_train))

    logging.info(f'Number of features: {NUM_FEATURES}; Number of classes: {NUM_CLASSES}')

    logging.info('train on the training set and predict on the validation and testing set...')

    if args.classifier == 'mlp':
        #// 函数用于计算 MLP 的维度
        mlp_dims = utils.get_model_dims('MLP', NUM_FEATURES, args.mlp_hidden, NUM_CLASSES)

        class_weight = None

        mlp_model_name = f'{args.data}_{args.classifier}_' + \
                         f'{mlp_dims}_lr{args.mlp_lr}_' + \
                         f'b{args.mlp_batch_size}_e{args.mlp_epochs}_' + \
                         f'd{args.mlp_dropout}.h5'

        MLP_MODEL_PATH = os.path.join(SAVED_MODEL_FOLDER, args.data, mlp_model_name)

        mlp_classifier = classifier.MLPClassifier(dims=mlp_dims,
                                                  model_save_name=MLP_MODEL_PATH,
                                                  dropout=args.mlp_dropout)

        # incase args.mlp_retrain = 0 while there is no Model file
        logging.debug(f'Saving MLP models to {MLP_MODEL_PATH}...')
        retrain_flag = 1 if not os.path.exists(MLP_MODEL_PATH) else args.mlp_retrain
        logging.debug(f'retrain? {retrain_flag}')
        
        #//训练 MLP 分类器,并返回验证集的准确率 val_acc
        val_acc = mlp_classifier.train(X_train, y_train,
                                       lr=args.mlp_lr,
                                       batch_size=args.mlp_batch_size,
                                       epochs=args.mlp_epochs,
                                       class_weight=class_weight,
                                       retrain=retrain_flag)

        saved_confusion_matrix_fig_path = os.path.join(FIG_FOLDER, args.data, 'intermediate', 'MLP_confusion_matrix.png')
        #//用于在测试集上进行预测，并计算测试集的准确率 new_acc 和混淆矩阵，并将混淆矩阵保存为一张图片。
        #//最后，函数会将预测结果 y_pred 和准确率 new_acc 返回给用户
        y_pred, new_acc = mlp_classifier.predict(X_test, y_test, args.data, args.newfamily_label,
                                                 saved_confusion_matrix_fig_path)

    elif args.classifier == 'rf':
        rf_model_name = f'{args.data}_{args.classifier}_{args.tree}.pkl'
        RF_MODEL_PATH = os.path.join(SAVED_MODEL_FOLDER, args.data, rf_model_name)
        rf_classifier = classifier.RFClassifier(RF_MODEL_PATH, args.tree)

        # incase args.rf_retrain = 0 while there is no Model file
        retrain_flag = 1 if not os.path.exists(RF_MODEL_PATH) else args.rf_retrain
        saved_confusion_matrix_fig_path = os.path.join(FIG_FOLDER, args.data, 'RF_confusion_matrix.png')
        #//模型在测试集上的准确率被存储到 val_acc 中，模型在新样本集上的准确率被存储到 new_acc 中
        y_pred, val_acc, new_acc = rf_classifier.fit_and_predict(X_train, y_train,
                                                                 X_test, y_test,
                                                                 args.data,
                                                                 args.newfamily_label,
                                                                 saved_confusion_matrix_fig_path,
                                                                 retrain=retrain_flag)
    else:
        logging.error(f'classifier {args.classifier} NOT supported.')
        sys.exit(-2)

    # --------------------------------------------------------- #
    # 4. Report the classification results                      #
    # --------------------------------------------------------- #
    logging.info('Report classification results for the wrongly classified and all the new samples...')

    name_tmp = f'{args.classifier}'
    # ALL: contains all the classification results in the testing set
    CLASSIFY_RESULTS_ALL_PATH = os.path.join(REPORT_FOLDER, f'{args.data}', 'intermediate',
                                             name_tmp + '_classification_results_all.csv')
    # SIMPLE: only contains misclassified samples in the testing set
    CLASSIFY_RESULTS_SIMPLE_PATH = os.path.join(REPORT_FOLDER, f'{args.data}', 'intermediate',
                                                name_tmp + '_classification_results_simple.csv')
    if args.classifier == 'mlp':
        if os.path.exists(CLASSIFY_RESULTS_ALL_PATH) and os.path.exists(CLASSIFY_RESULTS_SIMPLE_PATH) \
            and args.mlp_retrain == False:
            logging.debug(f'{CLASSIFY_RESULTS_ALL_PATH} and {CLASSIFY_RESULTS_SIMPLE_PATH} already exists.')
            pass
        else:
            evaluate.report_classification_results(MLP_MODEL_PATH,
                                                 X_test, y_test,
                                                 CLASSIFY_RESULTS_ALL_PATH,
                                                 CLASSIFY_RESULTS_SIMPLE_PATH)
    elif args.classifier == 'rf':
        if os.path.exists(CLASSIFY_RESULTS_ALL_PATH) and os.path.exists(CLASSIFY_RESULTS_SIMPLE_PATH) \
            and args.rf_retrain == False:
            logging.debug(f'{CLASSIFY_RESULTS_ALL_PATH} and {CLASSIFY_RESULTS_SIMPLE_PATH} already exists.')
            pass
        else:
            evaluate.report_classification_results(RF_MODEL_PATH,
                                                 X_test, y_test,
                                                 CLASSIFY_RESULTS_ALL_PATH,
                                                 CLASSIFY_RESULTS_SIMPLE_PATH)

    # --------------------------------------------------------- #
    # 5. Train the Contrastive Autoencoder                      #
    # --------------------------------------------------------- #
    #//基于对比损失函数的自编码器模型，用于学习数据的低维表示。
    logging.info('Training contrastive autoencoder...')
    #//获取对比自编码器的维度
    cae_dims = utils.get_model_dims('Contrastive AE', NUM_FEATURES,
                                    args.cae_hidden, NUM_CLASSES)
    #设定优化器为AdamOptimizer，模型保存路径为SAVED_MODEL_FOLDER和数据名称
    OPTIMIZER = tf.train.AdamOptimizer
    CKPT_DIR = os.path.join(SAVED_MODEL_FOLDER, args.data)
    utils.create_folder(CKPT_DIR)
    cae_dims_str = str(cae_dims).replace(' ', '').replace(',', '-').replace('[', '').replace(']', '') # remove extra symbols

    s1 = timer()
    #如果pure_ae参数为0，则使用对比自编码器进行训练；反之则使用普通自编码器进行训练
    if args.pure_ae == 0:
        ''' Our method: use contrastive autoencoder'''
        AE_WEIGHTS_PATH = os.path.join(CKPT_DIR, f'cae_{cae_dims_str}_lr{args.cae_lr}' + \
                                    f'_b{args.cae_batch_size}_e{args.cae_epochs}_m{args.margin}_lambda{args.cae_lambda_1}_weights.h5')
        cae = ContrastiveAE(cae_dims, OPTIMIZER, args.cae_lr)
        cae.train(X_train, y_train,
                args.cae_lambda_1, args.cae_batch_size, args.cae_epochs, args.similar_ratio, args.margin,
                AE_WEIGHTS_PATH, args.display_interval)
    #反之则使用普通自编码器进行训练
    else:
        '''baseline: use vanilla autoencoder'''
        AE_WEIGHTS_PATH = os.path.join(CKPT_DIR, f'pure_ae_{cae_dims_str}_lr{args.cae_lr}' + \
                                    f'_b{args.cae_batch_size}_e{args.cae_epochs}_m{args.margin}_weights.h5')
        pure_ae = Autoencoder(cae_dims)
        batch_size = int(args.cae_batch_size / 2) # CAE need the pair comparison, so we adjust it to half the CAE batch_size.
        pure_ae.train_and_save(X_train, AE_WEIGHTS_PATH, args.cae_lr, batch_size, args.cae_epochs, loss='mse')

    e1 = timer()
    logging.info(f'Training contrastive autoencoder time: {(e1 - s1):.3f} seconds')
    logging.info('Training contrastive autoencoder finished')

    # --------------------------------------------------------- #
    # 6. Detect drifting samples in the testing set                  #
    # --------------------------------------------------------- #
    ```
    logging.info('Detect drifting samples in the testing set...') # 记录日志，说明正在检测测试集中的漂移样本
    
    postfix_no_mad = f'm{args.margin}_lambda{args.cae_lambda_1}' # 根据 margin 和 cae_lambda_1 参数生成后缀名
    
    ALL_DETECT_PATH = os.path.join(REPORT_FOLDER, args.data, 'intermediate', # 生成保存所有测试集样本最近邻家族的文件路径
                                   f'{args.classifier}_detect_results_all_{postfix_no_mad}.csv')
    
    utils.create_parent_folder(ALL_DETECT_PATH) # 创建保存文件的父文件夹
    # 生成保存检测出的漂移样本的文件路径
    SIMPLE_DETECT_PATH = os.path.join(REPORT_FOLDER, args.data, 'intermediate', 
                                    f'{args.classifier}_detect_results_simple_{postfix_no_mad}.csv')

    TRAINING_INFO_FOR_DETECT_PATH = os.path.join(REPORT_FOLDER, args.data, 'intermediate', # 生成保存用于检测漂移样本所需信息的文件路径
                                                f'{args.classifier}_training_info_for_detect_{postfix_no_mad}.npz')

    s2 = timer()
    detect.detect_drift_samples(X_train, y_train, X_test, y_test, y_pred, # 检测测试集中的漂移样本
                                cae_dims,
                                args.margin,
                                args.mad_threshold,
                                AE_WEIGHTS_PATH,
                                ALL_DETECT_PATH, SIMPLE_DETECT_PATH,
                                TRAINING_INFO_FOR_DETECT_PATH)
    e2 = timer()
    logging.debug(f'detect_odd_samples time: {(e2 - s2):.3f} seconds') # 记录日志，说明检测漂移样本的时间
    logging.info('Detect drifting samples in the testing set finished') # 记录日志，说明已经完成检测测试集中的漂移样本
    # --------------------------------------------------------- #
    # 7. Evaluate the detection performance                     #
    # --------------------------------------------------------- #
    # 定义一个字符串变量，表示分类器和检测结果的文件路径
    name_tmp = f'{args.classifier}_combined_classify_detect_results_{postfix_no_mad}.csv'
    COMBINED_REPORT_PATH = os.path.join(REPORT_FOLDER, args.data, 'intermediate', name_tmp)

    # 调用 evaluate 模块中的函数，将分类器的结果和检测结果进行合并，并输出到 COMBINED_REPORT_PATH 文件中
    evaluate.combine_classify_and_detect_result(CLASSIFY_RESULTS_ALL_PATH, ALL_DETECT_PATH, COMBINED_REPORT_PATH)

    # 定义一个字符串变量，表示保存有序样本距离的文件路径
    SAVE_ORDERED_DIS_PATH = os.path.join(REPORT_FOLDER, args.data, 'intermediate',
                                         f'ordered_sample_real_mindis_{postfix_with_mad}.txt')

    # 定义一个字符串变量，表示保存距离、检测效率和 PR 值图的文件路径
    DIST_EFFORT_PR_VALUE_FIG_PATH = os.path.join(FIG_FOLDER, args.data,
                                                 f'dist_{args.classifier}_inspection_effort_pr_value_{postfix_with_mad}.png')

    # 定义一个字符串变量，表示保存逐个检查检测结果的文件路径
    DIST_ONE_BY_ONE_CHECK_RESULT_PATH = os.path.join(REPORT_FOLDER, args.data,
                                                     f'dist_{args.classifier}_one_by_one_check_pr_value_{postfix_with_mad}.csv')

    # 调用 evaluate 模块中的函数，评估新的家族是否为漂移样本，并输出相关结果到上述文件路径中
    evaluate.evaluate_newfamily_as_drift_by_distance(args.data, args.newfamily_label, COMBINED_REPORT_PATH, args.mad_threshold,
                                                     SAVE_ORDERED_DIS_PATH, DIST_EFFORT_PR_VALUE_FIG_PATH,
                                                     DIST_ONE_BY_ONE_CHECK_RESULT_PATH)

# 打印一条日志，表示检测性能评估已经完成
logging.info('Evaluate the detection performance finished')

    # --------------------------------------------------------- #
    # 8. Explain why it's an drifting sample                    #
    # --------------------------------------------------------- #
    ```
    if args.stage == 'explanation': # 检查是否处于解释阶段
            logging.info('Explain the detected drifting samples...') # 记录日志，说明正在解释漂移样本
            lambda1 = args.exp_lambda_1 # 解释方法的参数 lambda1
            exp_method = args.exp_method # 解释方法
            MASK_FILE_PATH = os.path.join(REPORT_FOLDER, args.data, f'mask_{exp_method}_{lambda1}.npz') # 生成掩码文件路径

            if exp_method == 'approximation_loose': # 如果是 approximation_loose 方法
                import cade.explain_global_approximation_loose_boundary as explain # 导入相关模块

                SAVED_EXP_CLASSIFIER_FOLDER = os.path.join(SAVED_MODEL_FOLDER, args.data, f'exp_{exp_method}') # 保存解释分类器的文件夹路径

                 # 使用漂移样本解释训练数据中每个实例的分类结果
                explain.explain_drift_samples_per_instance(X_train, y_train, X_test, y_test,
                                                           args,
                                                           DIST_ONE_BY_ONE_CHECK_RESULT_PATH,
                                                           TRAINING_INFO_FOR_DETECT_PATH,
                                                           AE_WEIGHTS_PATH,
                                                           SAVED_EXP_CLASSIFIER_FOLDER,
                                                           MASK_FILE_PATH)
            elif exp_method == 'distance_mm1': # 如果是 distance_mm1 方法
                '''explain by minimizing latent distance to centroid '''
                explain_dis.explain_drift_samples_per_instance(X_train, y_train, X_test, y_test, # 使用漂移样本解释训练数据中每个实例的分类结果
                                                               args,
                                                               DIST_ONE_BY_ONE_CHECK_RESULT_PATH,
                                                               TRAINING_INFO_FOR_DETECT_PATH,
                                                               AE_WEIGHTS_PATH,
                                                               MASK_FILE_PATH)
            else: # 如果是其他的方法
                logging.error(f'exp_method {exp_method} not supported') # 记录日志，说明该方法不受支持
                sys.exit(-3) # 退出程序

            logging.info('Explain the detected drifting samples finished') # 记录日志，说明漂移样本的解释已经完成
```


if __name__ == "__main__":
    start = timer()
    main()
    end = timer()
    logging.info(f'time elapsed: {end - start}')
