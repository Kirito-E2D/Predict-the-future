# -*- coding: utf-8 -*-
"""
Created on 2019/06/25
Updated on 2020/06/09
@author: Li Junyang
部分代码参考：https://blog.csdn.net
"""

'''
原始LSTM，输入4维：(时间、室外温度、室外湿度、1h前冷负荷)*24，
输出多维，加ANN
测试集工况包含在训练集内
timestep=24，loss用reduce_mean，学习率0.0001
有正则项，只对最后一个cell的输出计算loss
寻优超参数：隐状态神经元数、ANN激活函数
'''

import os
import time
import math
import xlsxwriter
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
#from sklearn import preprocessing
from tensorflow.python.keras import activations
from sklearn.model_selection import train_test_split


#——————————定义常量——————————
INPUT_SIZE = 4        #输入层维度
OUTPUT_SIZE = 1       #输出层维度
TIME_STEP = 24        #时间步（物理周期）
BATCH_SIZE = 500      #每一批次训练多少个样本
LEARNING_RATE = 0.0001 #学习率
REGULARIZER = 0.0001
hidden_layers = 2
MODEL_NAME = "6-10mon"
START = 3618-2-23
END = 7264-2
Max_ITER = 0          #最大训练次数
INER = 50    #每隔INER显示一次
CV_RMSE_best = 100


#——————————————————导入数据、填充缺失值——————————————————————
#读入数据
df = pd.read_csv('..\..\dataset\Office_Ayden.csv')    #数据集为1小时内取平均
df.fillna(method='bfill', inplace=True)
#df.interpolate(method='quadratic', axis=0, inplace=True)    #其他数据用二阶B样条插值
#df.interpolate(method='linear', axis=0, inplace=True)    #其他数据用线性插值 

#取出需要的一段数据
#df = df.iloc[START:START+TRAIN_NUM+TEST_NUM]
df = df.iloc[START:END+1]

#获取时间
Time = np.array(df['hour']).astype(int).reshape(-1)
#获取温度、湿度、1小时前冷负荷、时间、冷负荷
data = np.array(df.loc[:, ['TemperatureC', 'Humidity', 'energy 1h before', 'hour', 'energy']])

#max-min标准化
#minmax_scaler=preprocessing.MinMaxScaler()      #建立MinMaxScaler模型对象
#data_scale_2=minmax_scaler.fit_transform(data)  #MinMax标准化处理
CL_min = np.min(data[:, -1])
CL_max = np.max(data[:, -1])
normalized_data = (data-np.min(data,0))/(np.max(data,0)-np.min(data,0))

#Z-score标准化
#zscore_scaler = preprocessing.StandardScaler()     #建立StandarScaler对象
#normalized_data = zscore_scaler.fit_transform(data)   #StandardScaler标准化处理
#mean = np.mean(data[:,-1])
#std = np.std(data[:, -1])


#——————————————————生成训练集、验证集、测试集——————————————————————
##训练用数据集
#x_data, y_data = [], []
#Y_data = data[23:TRAIN_NUM+TIME_STEP-1, -1].reshape(-1, 1)    #用于展示训练集预测效果
#time_data = time[23:TRAIN_NUM+TIME_STEP-1]
#for i in range(TRAIN_NUM):    #TRAIN_NUM个训练样本
#    x = normalized_data[i:i+TIME_STEP, :-1]    #温度、湿度、1h前冷负荷、时间
#    y = normalized_data[i+TIME_STEP-1, -1, np.newaxis]    #冷负荷
#    x_data.append(x.tolist())
#    y_data.append(y.tolist())

#数据集
x_data, y_data = [], []
Y_data = data[23:, -1].reshape(-1, 1)    #数据集
time_data = Time[23:]
for i in range(END+1-START-23):    #个样本
    x = normalized_data[i:i+TIME_STEP, :-1]    #温度、湿度、1h前冷负荷、时间
    y = normalized_data[i+TIME_STEP-1, -1, np.newaxis]    #冷负荷
    x_data.append(x.tolist())
    y_data.append(y.tolist())
    
#划分训练集、验证集和测试集
x_train, x_rest, y_train, y_rest = train_test_split(x_data,y_data,test_size=0.3, random_state=1)
time_train, time_rest, _, _ = train_test_split(time_data,y_data,test_size=0.3, random_state=1)
Y_train, Y_rest, _, _ = train_test_split(Y_data,y_data,test_size=0.3, random_state=1)

x_valid, x_test, y_valid, y_test = train_test_split(x_rest,y_rest,test_size=0.5, random_state=1)
time_valid, time_test, _, _ = train_test_split(time_rest,y_rest,test_size=0.5, random_state=1)
Y_valid, Y_test, _, _ = train_test_split(Y_rest,y_rest,test_size=0.5, random_state=1)


#——————————————————定义LSTM结构——————————————————
def lstm(hidden_unit, activation):
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_unit, forget_bias=1.0, state_is_tuple=True)
    #init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn,final_states = tf.nn.dynamic_rnn(cell, X, initial_state=None,
                                              dtype=tf.float32, time_major=False)  
    #output_rnn记录lstm每个time_step的输出，final_states是最后一个cell的状态
    output = tf.transpose(output_rnn, [1,0,2])    #交换batch_size和time_step
    output = output[-1]    #取出最后一个step的输出作为全连接层的输入
    #output = tf.reshape(output_rnn,[-1,HIDDEN_UNIT])    #作为MLP的输入
    activate = activations.get(activation)    #取出ANN的激活函数
    w_out1 = weights['out1']
    b_out1 = biases['out1']
    y1 = activate(tf.matmul(output, w_out1) + b_out1)
    w_out2 = weights['out2']
    b_out2 = biases['out2']
    y2 = activate(tf.matmul(y1, w_out2) + b_out2)
    w_out3 = weights['out3']
    b_out3 = biases['out3']
    pred = tf.matmul(y2, w_out3) + b_out3    #[batch_size, output_size]
    return pred


#——————————————————训练模型——————————————————
def train_lstm(batch_size, HIDDEN_UNIT, activation, gpu_options, Max_ITER):
    pred = lstm(HIDDEN_UNIT, activation)
    pre_step = 0
    #损失函数
    #loss = tf.reduce_sum(tf.square(tf.reshape(Y,[-1])-tf.reshape(pred,[-1]))) / tf.reduce_sum(np.square(tf.reshape(Y,[-1])-tf.reduce_mean(tf.reshape(Y,[-1]))))
    loss = tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y,[-1]))) + tf.add_n(tf.get_collection('losses'))    #tf.add_n用于'losses'中的元素相加
    #loss = tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y,[-1])))
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    saver = tf.train.Saver(max_to_keep=15)
    Num_of_iter = []
    train_loss = []
    valid_loss = []
    with tf.Session() as sess:
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        #sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            pre_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])

        #重复训练ITER次
        for step in range(1, Max_ITER+1):
            start = 0
            end = start+batch_size
            while(end<len(x_train)):
                sess.run(train_op, feed_dict={X:x_train[start:end],
                                              Y:y_train[start:end]})
                start += batch_size
                end = start+batch_size
            if step % INER == 0:
                Num_of_iter.append((pre_step+step))    #训练次数
                train_loss_ = sess.run(loss, feed_dict={X:x_train,Y:y_train})
                train_loss.append(train_loss_.tolist())
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=pre_step+step)
                valid_loss_ = sess.run(loss, feed_dict={X:x_valid,Y:y_valid})
                valid_loss.append(valid_loss_.tolist()) 
                print('After %d training step(s), loss on training set is %g, loss on validation set is %g.' % (pre_step+step, train_loss_, valid_loss_))
            if len(valid_loss) > 1:
                if ((valid_loss[-1]-valid_loss[-2]) > 0) or ((valid_loss[-2]-valid_loss[-1]) / valid_loss[-2] < 0.0008):
                    print('Training process is done.')
                    break
                
        predict_valid = sess.run(pred, feed_dict={X:x_valid})
        predictions_valid = np.array(predict_valid)*(CL_max - CL_min) + CL_min
        #predictions_valid = np.array(predict_valid)*std + mean
        RMSE = np.sqrt(np.mean(np.square(Y_valid-predictions_valid)))    #均方根误差
        CV_RMSE = RMSE / np.mean(Y_valid)

            
    '''
    #输出损失变化
    workbook = xlsxwriter.Workbook('./Loss of' + FIG_NAME + '_' + MODEL_NAME + '_' + str((Num_of_iter[-1] if len(Num_of_iter)>0 else pre_step)) + '.xlsx')
    worksheet = workbook.add_worksheet('Loss')
    worksheet.write(0, 0, 'Number of iterations')
    worksheet.write(0, 1, 'Train Loss')
    #worksheet.write(0, 2, 'Validation Loss')
    for i in range(len(train_loss)):
        worksheet.write(i+1, 0, Num_of_iter[i])
        worksheet.write(i+1, 1, train_loss[i])
        #worksheet.write(i+1, 2, val_loss[i])
    workbook.close()
    '''
    
       
#    #画图
#    plt.close('all')    #关闭之前画的图
#    plt.figure(num='Loss', figsize=(6.4*1.5,4.8*1.5), dpi=100)
#    pos = np.arange(0, len(train_loss), (int(len(train_loss)/25) if int(len(train_loss)/25)>0 else 1))
#    x = np.arange(1, len(pos)+1)
#    plt.plot(x, np.array(train_loss)[pos], 'b-^', label='Train loss')
#    plt.plot(x, np.array(valid_loss)[pos], 'r-s', label='Validation loss')
#    plt.xticks(x, np.array(Num_of_iter)[pos], rotation=80)
#    plt.xlabel('Number of iterations', fontsize=15)
#    plt.ylabel('Loss', fontsize=15)
#    plt.legend(fontsize=20)
#    #plt.savefig('Loss of ' + FIG_NAME + '_' + MODEL_NAME +'_' + str((Num_of_iter[-1] if len(Num_of_iter)>0 else pre_step)) + '.png')
#    plt.show()
    
    return CV_RMSE

tf.reset_default_graph()
gpu_options = tf.GPUOptions(allow_growth=True)
start = time.perf_counter()    #计算程序运行时间
for HIDDEN_UNIT in [5]:
    for activation in ['tanh']:
        with tf.variable_scope('train_' + str(HIDDEN_UNIT) + '_' + activation):
            print('The number of hidden units is ' + str(HIDDEN_UNIT) + ', the activation is ' + activation)
            MODEL_SAVE_PATH = './model_V0.3_4_24_' + str(HIDDEN_UNIT) + '_' + str(hidden_layers) + '_' + activation + '_regu_in/'
            FIG_NAME = 'V0.3_4_24_' + str(HIDDEN_UNIT) + '_' + str(hidden_layers) + '_' + activation + '_regu_in'
    
            #——————————————————定义神经网络变量——————————————————
            X = tf.placeholder(tf.float32, [None,TIME_STEP,INPUT_SIZE])    #每批次输入网络的tensor
            Y = tf.placeholder(tf.float32, [None,OUTPUT_SIZE])   #每批次tensor对应的标签

            #DNN输出层权重、偏置
            weights={
                    'out1':tf.Variable(tf.random_normal([HIDDEN_UNIT, math.ceil(HIDDEN_UNIT*2/3+1)])),
                    'out2':tf.Variable(tf.random_normal([math.ceil(HIDDEN_UNIT*2/3+1), math.ceil(HIDDEN_UNIT*2/3+1)])),
                    'out3':tf.Variable(tf.random_normal([math.ceil(HIDDEN_UNIT*2/3+1) ,OUTPUT_SIZE]))
                    }
            biases={
                    'out1':tf.Variable(tf.constant(0.1,shape=[math.ceil(HIDDEN_UNIT*2/3+1),])),
                    'out2':tf.Variable(tf.constant(0.1,shape=[math.ceil(HIDDEN_UNIT*2/3+1),])),
                    'out3':tf.Variable(tf.constant(0.1,shape=[OUTPUT_SIZE,]))
                    }
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZER)(weights['out1']))
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZER)(weights['out2']))
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZER)(weights['out3']))    
    
    
            CV_RMSE = train_lstm(BATCH_SIZE, HIDDEN_UNIT, activation, gpu_options, Max_ITER)
            if CV_RMSE < CV_RMSE_best:
                CV_RMSE_best = CV_RMSE
                MODEL_SAVE_PATH_best = MODEL_SAVE_PATH
                FIG_NAME_best = FIG_NAME 
                HIDDEN_UNIT_best = HIDDEN_UNIT
                ACTIVATION_best = activation
                
print('The best number of hidden units is ' + str(HIDDEN_UNIT_best) + ', the best activation is ' + ACTIVATION_best)
print('the best CVRMSE is ' + str(CV_RMSE_best))

end = time.perf_counter()
t = end - start
print("Training time is ：", t)


##使用最佳参数训练一下，让预测部分的代码不出错
##tf.reset_default_graph()
#with tf.variable_scope('train_' + str(HIDDEN_UNIT_best) + '_' + ACTIVATION_best, reuse=True):
#    print('The best number of hidden units is ' + str(HIDDEN_UNIT_best) + ', the best activation is ' + ACTIVATION_best)
#    MODEL_SAVE_PATH = './model_V0.3_4_24_' + str(HIDDEN_UNIT_best) + '_1_' + ACTIVATION_best + '_regu_last_mean_valid_general/'
#    FIG_NAME = 'V0.3_4_24_' + str(HIDDEN_UNIT_best) + '_1_' + ACTIVATION_best + '_regu_last_mean_valid_general'
#    
#    #——————————————————定义神经网络变量——————————————————
#    X = tf.placeholder(tf.float32, [None,TIME_STEP,INPUT_SIZE])    #每批次输入网络的tensor
#    Y = tf.placeholder(tf.float32, [None,OUTPUT_SIZE])   #每批次tensor对应的标签
#
#    #DNN输出层权重、偏置
#    weights={
#            'out1': tf.Variable(tf.random_normal([HIDDEN_UNIT_best, math.ceil(HIDDEN_UNIT_best*2/3+1)])),
#            'out2': tf.Variable(tf.random_normal([math.ceil(HIDDEN_UNIT_best*2/3+1), OUTPUT_SIZE]))
#            }
#    biases={
#            'out1': tf.Variable(tf.constant(0.1,shape=[math.ceil(HIDDEN_UNIT_best*2/3+1),])),
#            'out2': tf.Variable(tf.constant(0.1,shape=[OUTPUT_SIZE,]))
#            }
#    
#    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZER)(weights['out1']))
#    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZER)(weights['out2']))
#    
#    CV_RMSE = train_lstm(BATCH_SIZE, HIDDEN_UNIT_best, ACTIVATION_best, gpu_options, 10)


#————————————————预测模型————————————————————
def prediction(MODEL_SAVE_PATH_best, FIG_NAME_best, HIDDEN_UNIT_best, activation, gpu_options):
    pred = lstm(HIDDEN_UNIT_best, activation)      #预测时输入[none,time_step,input_size]的验证数据
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH_best)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)    #参数恢复
            total_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            
            
#            #展示训练集预测效果
#            predict_train = sess.run(pred, feed_dict={X:x_train})
#            
#            #计算训练集误差
#            #predictions_train = predict_train
#            predictions_train = np.array(predict_train)*(CL_max - CL_min) + CL_min
#            #predictions_train = np.array(predict_train)*std + mean
#            error_train = Y_train-predictions_train
#            MAE = np.mean(abs(Y_train-predictions_train))    #绝对误差
#            #MSE = np.mean(np.square(y_train-predictions_train))    #均方误差
#            RMSE = np.sqrt(np.mean(np.square(Y_train-predictions_train)))    #均方根误差
#            CV_RMSE = RMSE / np.mean(abs(Y_train))
#            MAPE = np.mean(abs(Y_train-predictions_train)/abs(Y_train))    #相对百分比误差
#            #CPGE = abs(Y_train.cumsum()-predictions_train.cumsum()) / Y_train.cumsum() * 100    #累积冷负荷误差
#            R2 = 1 - np.sum(np.square(Y_train-predictions_train)) / np.sum(np.square(Y_train-np.mean(Y_data)))
#            print('After %s training step(s), MAE on train set is %g, RMSE on train set is %g, MAPE on train set is %g, CV_RMSE on train set is %g, R2 on train set is %g.' %(total_step, MAE, RMSE, MAPE, CV_RMSE, R2))
#            #print(MSE)
#            
#            #以折线图表示训练集结果
#            x = range(len(Y_train[:100]))
#            names = time_train[:100].tolist()[::2]
#            pos = x[::2]
#            plt.figure(figsize=(6.4*3.2,4.8*3.2), dpi=150)
#            plt.plot(x, Y_train[:100], 'b-', label='Actual value of trainset')
#            plt.plot(x, predictions_train[:100], 'r--', label='Forecasting value')
#            plt.xticks(pos, names, rotation=80)
#            plt.tick_params(labelsize=15)
#            plt.xlabel('Time', fontsize=20)
#            plt.ylabel('Cooling load(kW)', fontsize=20)
#            plt.legend(fontsize=20)
#            #plt.savefig('Prediction of train' + '_' + FIG_NAME_best + '_' + MODEL_NAME + '.png')
#            plt.show()
#            
#            workbook = xlsxwriter.Workbook('./Prediction and error of data.xlsx')
#            worksheet = workbook.add_worksheet('Prediction error')
#            worksheet.write(0, 0, 'prediction')
#            worksheet.write(0, 1, 'error')
#            for i in range(len(y_train)):
#                worksheet.write(i+1, 0, predictions_train[i])
#                worksheet.write(i+1, 1, error_train[i])
#                worksheet.write(i+1, 2, predict_train[i])
#                
#            workbook.close()
#            
#            
#            #展示验证集预测效果
#            predict_valid = sess.run(pred, feed_dict={X:x_valid})
#            
#            #计算验证集误差
#            predictions_valid = np.array(predict_valid)*(CL_max - CL_min) + CL_min
#            #predictions_train = np.array(predict_train)*std + mean
#            error_valid = Y_valid - predictions_valid
#            MAE = np.mean(abs(Y_valid-predictions_valid))    #绝对误差
#            #MSE = np.mean(np.square(y_train-predictions_train))    #均方误差
#            RMSE = np.sqrt(np.mean(np.square(Y_valid-predictions_valid)))    #均方根误差
#            CV_RMSE = RMSE / np.mean(abs(Y_valid))
#            MAPE = np.mean(abs(Y_valid-predictions_valid)/abs(Y_valid))    #相对百分比误差
#            R2 = 1 - np.sum(np.square(Y_valid-predictions_valid)) / np.sum(np.square(Y_valid-np.mean(Y_valid)))
#            print('After %s training step(s), MAE on train set is %g, RMSE on train set is %g, MAPE on train set is %g, CV_RMSE on train set is %g, R2 on train set is %g.' %(total_step, MAE, RMSE, MAPE, CV_RMSE, R2))
#            #print(MSE)
#            
#            #以折线图表示训练集结果
#            x = range(len(Y_valid[:100]))
#            names = time_valid[:100].tolist()[::2]
#            pos = x[::2]
#            plt.figure(figsize=(6.4*3.2,4.8*3.2), dpi=150)
#            plt.plot(x, Y_valid[:100], 'b-', label='Actual value of validation')
#            plt.plot(x, predictions_valid[:100], 'r--', label='Forecasting value')
#            plt.xticks(pos, names, rotation=80)
#            plt.tick_params(labelsize=15)
#            plt.xlabel('Time', fontsize=20)
#            plt.ylabel('Error(kW)', fontsize=20)
#            plt.legend(fontsize=20)
#            #plt.savefig('Prediction of train' + '_' + FIG_NAME_best + '_' + MODEL_NAME + '.png')
#            plt.show()
#            
#            workbook = xlsxwriter.Workbook('./Prediction and error of validation.xlsx')
#            worksheet = workbook.add_worksheet('Prediction error')
#            worksheet.write(0, 0, 'prediction')
#            worksheet.write(0, 1, 'error')
#            for i in range(len(y_train)):
#                worksheet.write(i+1, 0, predictions_train[i])
#                worksheet.write(i+1, 1, error_valid[i])
#                worksheet.write(i+1, 2, predict_valid[i])
#            workbook.close()
            
            
#            #展示整个数据集预测效果
#            predict_data = sess.run(pred, feed_dict={X:x_data})
#            
#            #计算训练集误差
#            predictions_data = np.array(predict_data)*(CL_max - CL_min) + CL_min
#            #predictions_data = np.array(predict_train)*std + mean
#            error_data = Y_data - predictions_data
#            MAE = np.mean(abs(Y_data-predictions_data))    #绝对误差
#            #MSE = np.mean(np.square(y_train-predictions_train))    #均方误差
#            RMSE = np.sqrt(np.mean(np.square(Y_data-predictions_data)))    #均方根误差
#            CV_RMSE = RMSE / np.mean(abs(Y_data))
#            MAPE = np.mean(abs(Y_data-predictions_data)/abs(Y_data))    #相对百分比误差
#            #CPGE = abs(Y_data.cumsum()-predictions_train.cumsum()) / Y_data.cumsum() * 100    #累积冷负荷误差
#            R2 = 1 - np.sum(np.square(Y_data-predictions_data)) / np.sum(np.square(Y_data-np.mean(Y_data)))
#            print('After %s training step(s), MAE on train set is %g, RMSE on train set is %g, MAPE on train set is %g, CV_RMSE on train set is %g, R2 on train set is %g.' %(total_step, MAE, RMSE, MAPE, CV_RMSE, R2))
#            #print(MSE)
#            
#            #以折线图表示训练集结果
#            x = range(len(Y_data[:100]))
#            names = time_data[:100].tolist()[::2]
#            pos = x[::2]
#            plt.figure(figsize=(6.4*3.2,4.8*3.2), dpi=150)
#            plt.plot(x, Y_data[:100], 'b-', label='Actual value of dataset')
#            plt.plot(x, predictions_data[:100], 'r--', label='Forecasting value')
#            plt.xticks(pos, names, rotation=80)
#            plt.tick_params(labelsize=15)
#            plt.xlabel('Time', fontsize=20)
#            plt.ylabel('Cooling load(kW)', fontsize=20)
#            plt.legend(fontsize=20)
#            #plt.savefig('Prediction of train' + '_' + FIG_NAME_best + '_' + MODEL_NAME + '.png')
#            plt.show()
#            
#            workbook = xlsxwriter.Workbook('./Prediction and error of dataset.xlsx')
#            worksheet = workbook.add_worksheet('Prediction and error')
#            worksheet.write(0, 0, 'prediction')
#            worksheet.write(0, 1, 'error')
#            for i in range(len(y_train)):
#                worksheet.write(i+1, 0, predictions_data[i])
#                worksheet.write(i+1, 1, error_data[i])
#            workbook.close()
            
            
            #展示测试集预测效果
            predict = sess.run(pred, feed_dict={X:x_test})
            
            #计算测试集误差
            #predictions = predict
            predictions = np.array(predict)*(CL_max - CL_min) + CL_min
            #predictions = np.array(predict)*std + mean
            error = Y_test - predictions
            MAE = np.mean(abs(Y_test-predictions))    #绝对误差
            RMSE = np.sqrt(np.mean(np.square(Y_test-predictions)))    #均方根误差
            CV_RMSE = RMSE / np.mean(abs(Y_test))
            MAPE = np.mean(abs(Y_test-predictions)/(abs(Y_test)))    #相对百分比误差
            #CPGE = abs(predictions.cumsum()-Y_test.cumsum()) / Y_test.cumsum() * 100    #累积冷负荷误差
            R2 = 1 - np.sum(np.square(Y_test-predictions)) / np.sum(np.square(Y_test-np.mean(Y_test)))
            print('After %s training step(s), MAE on test set is %g, RMSE on test set is %g, MAPE on test set is %g, CV_RMSE on test set is %g, R2 on test set is %g.' %(total_step, MAE, RMSE, MAPE, CV_RMSE, R2))
            
            
            #以折线图表示测试集结果
            x = range(len(Y_test[:100]))
            names = time_test[:100].tolist()[::2]
            pos = x[::2]
            
            '''
            plt.figure(figsize=(6.4*3.2,4.8*3.2), dpi=150)
            plt.plot(x, CPGE[:100], 'b-')
            plt.xticks(pos, names, rotation=80)
            plt.tick_params(labelsize=15)
            plt.xlabel('Time', fontsize=20)
            plt.ylabel('CPGE(%)', fontsize=20)
            plt.savefig('CPGE of test' + '_' + FIG_NAME_best + '_' + MODEL_NAME + '.png')
            plt.show()
            '''
            
            plt.figure(figsize=(6.4*3.2,4.8*3.2), dpi=150)
            plt.plot(x, Y_test[:100], 'b-', label='Actual value')
            plt.plot(x, predictions[:100], 'r--', label='Forecasting value')
            plt.xticks(pos, names, rotation=80)
            plt.tick_params(labelsize=15)
            plt.xlabel('Time', fontsize=20)
            plt.ylabel('Cooling load(kW)', fontsize=20)
            plt.legend(fontsize=20)
            #plt.savefig('Prediction of test' + '_' + FIG_NAME_best + '_' + MODEL_NAME + '.png')
            plt.show()
            
#            #输出预测误差
#            workbook = xlsxwriter.Workbook('./Prediction and error of test of ' + FIG_NAME + '_' + MODEL_NAME + '.xlsx')
#            worksheet = workbook.add_worksheet('Prediction error')
#            worksheet.write(0, 0, 'prediction')            
#            worksheet.write(0, 1, 'prediction')
#            worksheet.write(0, 2, 'error')
#            for i in range(predictions.shape[0]):
#                worksheet.write(i+1, 0, Y_test[i])
#                worksheet.write(i+1, 1, predictions[i])
#                worksheet.write(i+1, 2, error[i])
#            workbook.close()
            
            '''
            #输出预测结果
            workbook = xlsxwriter.Workbook('./Prediction of test' + FIG_NAME + '_' + MODEL_NAME + '.xlsx')
            worksheet = workbook.add_worksheet('Prediction')
            worksheet.write(0, 0, 'Time')
            worksheet.write(0, 1, 'Real data')
            worksheet.write(0, 2, 'Prediction data')
            worksheet.write(0, 3, 'MAE of prediction')
            worksheet.write(0, 4, 'RMSE of prediction')
            worksheet.write(0, 5, 'MAPE of prediction')
            worksheet.write(0, 6, 'CPGE of prediction')
            worksheet.write(0, 7, 'R2 of prediction')
            worksheet.write(0, 8, 'CV_RMSE of prediction')
            worksheet.write(1, 3, MAE)
            worksheet.write(1, 4, RMSE)
            worksheet.write(1, 5, MAPE)
            worksheet.write(1, 7, R2)
            worksheet.write(1, 8, CV_RMSE)
            for i in range(predictions.shape[0]):
                worksheet.write(i+1, 0, time_test[i])
                worksheet.write(i+1, 1, Y_test[i])
                worksheet.write(i+1, 2, predictions[i])
                worksheet.write(i+1, 6, CPGE[i])
            workbook.close()
            '''
            
        else:
            print('No checkpoint file found')
#    return predictions
    return error


with tf.variable_scope('train_' + str(HIDDEN_UNIT_best) + '_' + ACTIVATION_best, reuse=True):
    start = time.perf_counter()    #计算程序运行时间
    final_pred = prediction(MODEL_SAVE_PATH_best, FIG_NAME_best, HIDDEN_UNIT_best, ACTIVATION_best, gpu_options)
    end = time.perf_counter()
    t = end - start
    print("Prediction time is ：", t)