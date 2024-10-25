#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2023 - 2023 ZhangPulin, Inc. All Rights Reserved 
#
# @Time    : 2023/11/21 19:26
# @Author  : ZhangPulin
# @File    : model_build_multimode.py
# @IDE     : PyCharm

import matplotlib.pylab as plt
import pandas as pd
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import MultiHeadAttention
from keras.metrics import Accuracy
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import *
from sklearn.metrics import mean_absolute_percentage_error

class PEMFC_model():
    def __init__(self):
        self.data_dir = './data/model_data.csv'
        raw_data = pd.read_csv(self.data_dir)
        self.scalar = MinMaxScaler()
        self.time_stride = 1
        self.time_read = 40
        self.time_predict = 1
        self.batchsize = 16
        self.epoch = int(200)
        self.density_model_epoch=int(1e2)
        self.optimizer = Adam(0.001, 0.5)
        self.num_nodes = 40
        self.name = 'multimode'##['humidity','temperature','density','voltage','hfr','multimode']
        self.version=3.2
        self.mode = 'multimode'##['singlemode','multimode']
        self.dict = {'humidity':6,'temperature':1,'density':0}
        mape = tf.keras.losses.MeanAbsolutePercentageError()
        ca_edgs = [(0,5),(5,10),(10,15),(15,20),(20,25),(25,30),(30,35),(1,6),(6,11),(11,16),(16,21),(21,26),(26,31),
                   (31,36),(2,7),(7,12),(12,17),(17,22),(22,27),(27,32),(32,37),(3,8),(8,13),(13,18),(18,23),(23,28),
                   (28,33),(33,38),(4,9),(9,14),(14,19),(19,24),(24,29),(29,34),(34,39)]
        an_edgs = ca_edgs
        liquid_edgs = [(4,3),(4,9),(3,2),(3,8),(2,1),(2,7),(1,0),(1,6),(0,5),(9,14),(9,8),(8,13),(8,7),(7,12),(7,6),(6,11),(6,5),
                       (5,10),(14,19),(14,13),(13,18),(13,12),(12,17),(12,11),(11,16),(11,10),(10,15),(19,24),(19,18),(18,23),(18,17),
                       (17,22),(17,16),(16,21),(16,15),(15,20),(24,29),(24,23),(23,28),(23,22),(22,27),(22,21),(21,26),(21,20),(20,25),
                       (29,34),(29,28),(28,33),(28,27),(27,32),(27,26),(26,31),(26,25),(25,30),(34,39),(34,33),(33,38),(33,32),(32,37),
                       (32,31),(31,36),(31,30),(30,35)]

        self.ca_adj = self.adj_build(ca_edgs)
        self.an_adj = self.adj_build(an_edgs)
        self.liquid_adj = self.adj_build(liquid_edgs)

        ### data processing
        ## density epoch=100 acc=90; hfr epoch=100 acc=50; voltage epoch=
        ## X_in: ['density_load','temperature_water_in','an_RH','an_ER','an_P','ca_P','ca_RH','ca_ER']
        train_X_his,train_env_his,train_X_in,train_Y,test_X_his,test_env_his,test_X_in,test_Y = self.data_processing(raw_data)
        t_density,t_y_density,te_density,te_y_density = train_X_his[:,:,:40],train_Y[:,:,:40],test_X_his[:,:,:40],test_Y[:,:,:40]
        t_voltage,t_y_voltage,te_voltage,te_y_voltage = train_X_his[:,:,40:80],train_Y[:,:,40:80],test_X_his[:,:,40:80],test_Y[:,:,40:80]
        t_hfr,t_y_hfr,te_hfr,te_y_hfr = train_X_his[:,:,80:120],train_Y[:,:,80:120],test_X_his[:,:,80:120],test_Y[:,:,80:120]
        t_humidity,t_y_humidity,te_humidity,te_y_humidity = train_X_his[:,:,120:160],train_Y[:,:,120:160],test_X_his[:,:,120:160],test_Y[:,:,120:160]
        t_temperature,t_y_temperature,te_temperature,te_y_temperature = train_X_his[:,:,160:200],train_Y[:,:,160:200],test_X_his[:,:,160:200],test_Y[:,:,160:200]

        self.feature_num = 40
        self.total_num = train_X_his.shape[-1]
        self.env_feature_num = train_X_in.shape[-1]
        density_model = load_model('./{}_Model/{}_Model_epoch{}.h5'.format('density', 'density', self.density_model_epoch))
        with open('./{}_Model/epoch{}_max_data.txt'.format('density',self.density_model_epoch), "r") as file:
            density_max = float(file.read())
        ### model build

        t_max = np.max(t_temperature)
        t_min = np.min(t_temperature)
        with open('./{}_Model/epoch{}_max_data.txt'.format('temperature', self.epoch), "w") as file:
            file.write(str(t_max))
        with open('./{}_Model/epoch{}_min_data.txt'.format('temperature', self.epoch), "w") as file:
            file.write(str(t_min))
        t_max_min =t_max-t_min
        t_temperature=(t_temperature-t_min)/t_max_min
        te_temperature=(te_temperature-t_min)/t_max_min
        t_y_temperature=np.clip((t_y_temperature-t_min)/t_max_min,1e-3,1)
        te_y_temperature=np.clip((te_y_temperature-t_min)/t_max_min,1e-3,1)
        t_id = self.dict['temperature']
        train_env_his[:,:,t_id],train_X_in[:,:,t_id],test_env_his[:,:,t_id],test_X_in[:,:,t_id]=\
            (train_env_his[:,:,t_id]-t_min)/t_max_min,(train_X_in[:,:,t_id]-t_min)/t_max_min,(test_env_his[:,:,t_id]-t_min)/t_max_min,(test_X_in[:, :, t_id]-t_min)/t_max_min

        h_max = np.max(t_humidity)
        h_min = np.min(t_humidity)
        with open('./{}_Model/epoch{}_max_data.txt'.format('humidity', self.epoch), "w") as file:
            file.write(str(h_max))
        with open('./{}_Model/epoch{}_min_data.txt'.format('humidity', self.epoch), "w") as file:
            file.write(str(h_min))
        h_max_min = h_max-h_min
        t_humidity=(t_humidity-h_min)/h_max_min
        te_humidity=(te_humidity-h_min)/h_max_min
        t_y_humidity=np.clip((t_y_humidity-h_min)/h_max_min,1e-3,1)
        te_y_humidity=np.clip((te_y_humidity-h_min)/h_max_min,1e-3,1)
        h_id = self.dict['humidity']
        train_env_his[:,:,h_id],train_X_in[:,:,h_id],test_env_his[:,:,h_id],test_X_in[:,:,h_id]=\
            (train_env_his[:,:,h_id]-h_min)/h_max_min,(train_X_in[:,:,h_id]-h_min)/h_max_min,(test_env_his[:,:,h_id]-h_min)/h_max_min,(test_X_in[:,:,h_id]-h_min)/h_max_min

        d_max = np.max(t_density)
        d_min = np.min(t_density)
        with open('./{}_Model/epoch{}_max_data.txt'.format('density', self.epoch), "w") as file:
            file.write(str(d_max))
        with open('./{}_Model/epoch{}_min_data.txt'.format('density', self.epoch), "w") as file:
            file.write(str(d_min))
        t_density=t_density/d_max
        te_density=te_density/d_max
        d_id = self.dict['density']
        train_env_his[:,:,d_id],train_X_in[:,:,d_id],test_env_his[:,:,d_id],test_X_in[:,:,d_id]=\
            train_env_his[:,:,d_id]/d_max,train_X_in[:,:,d_id]/d_max,test_env_his[:,:,d_id]/d_max,test_X_in[:,:,d_id]/d_max

        hfr_max = np.max(t_hfr)
        hfr_min = np.min(t_hfr)
        with open('./{}_Model/epoch{}_max_data.txt'.format('hfr', self.epoch), "w") as file:
            file.write(str(hfr_max))
        with open('./{}_Model/epoch{}_min_data.txt'.format('hfr', self.epoch), "w") as file:
            file.write(str(hfr_min))
        t_hfr = (t_hfr-hfr_min)/(hfr_max-hfr_min)
        te_hfr = (te_hfr-hfr_min)/(hfr_max-hfr_min)
        t_y_hfr = np.clip((t_y_hfr-hfr_min)/(hfr_max-hfr_min),1e-3,1)
        te_y_hfr = np.clip((te_y_hfr-hfr_min)/(hfr_max-hfr_min),1e-3,1)

        t_density_predict = density_model.predict([t_density,train_env_his,train_X_in])*density_max
        te_density_predict = density_model.predict([te_density,test_env_his,test_X_in])*density_max

        pemfc_model = self.PEMFC_build()
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        pemfc_model.compile(loss='mse', optimizer=self.optimizer, metrics=['mape'])
        history=pemfc_model.fit(x=[t_temperature,t_humidity,t_hfr,t_density_predict,train_env_his,train_X_in],
            y=[t_y_temperature,t_y_humidity,t_y_hfr], validation_data=([te_temperature,te_humidity,te_hfr,te_density_predict,test_env_his,test_X_in],
                    [te_y_temperature,te_y_humidity,te_y_hfr]),epochs=self.epoch, batch_size=self.batchsize,verbose=2)
        pemfc_model.save('./{}_Model/{}_Model{}_epoch{}.h5'.format(self.name,self.name,self.version,self.epoch))
        for name in ['Temperature','Humidity','Hfr']:
            self.plot_history(history,name)

    def data_processing(self,rawdata):
        data_index_nums=rawdata.last_valid_index()+1
        # scalar_data = self.scalar.fit_transform(rawdata)
        scalar_data = rawdata.values
        time_range = self.time_read+self.time_predict
        sample_num = (data_index_nums-time_range)//self.time_stride
        dataset = np.zeros((sample_num,time_range,rawdata.shape[-1]))
        for i in range(0,data_index_nums-time_range-1,self.time_stride):
            dataset[i//self.time_stride,:,:]=scalar_data[i:i+time_range,:]
        trainset,testset = train_test_split(dataset,test_size=0.2,random_state=12)
        trainset_X_his = trainset[:,:self.time_read,1:201]
        trainset_env_his = trainset[:,:self.time_read,201:]
        trainset_X_in = trainset[:,self.time_read:,201:]
        trainset_Y = trainset[:,self.time_read:,1:201]
        testset_X_his = testset[:,:self.time_read,1:201]
        testset_env_his = testset[:,:self.time_read,201:]
        testset_X_in = testset[:, self.time_read:,201:]
        testset_Y = testset[:,self.time_read:,1:201]
        return trainset_X_his, trainset_env_his,trainset_X_in, trainset_Y, testset_X_his, testset_env_his,testset_X_in, testset_Y

    def adj_build(self,edges):
        # 创建一个空的稀疏邻接矩阵 (Compressed Sparse Row format)
        adjacency_matrix = csr_matrix((self.num_nodes, self.num_nodes), dtype=int)
        # 添加边，根据节点连接关系构建稀疏邻接矩阵
        for edge in edges:
            # 在节点之间添加边
            adjacency_matrix[edge[0], edge[1]] = 1
            # 无向图的话，也要添加反向边
            adjacency_matrix[edge[1], edge[0]] = 1
        # 返回邻接矩阵
        return adjacency_matrix.toarray()

    def PEMFC_build(self):
        ### Temperature_encode model
        Temperature_his_data = Input(shape=(self.time_read, self.feature_num))
        env_his_data = Input(shape=(self.time_read,self.env_feature_num))
        env_data = Input(shape=(self.time_predict,self.env_feature_num))
        T_id = int(self.dict['temperature'])
        T_env_his = env_his_data[:,:,T_id]
        T_env_his = tf.tile(tf.expand_dims(T_env_his, axis=-1),[1,1,40])
        T_single = env_data[:,:,T_id]
        T_single = tf.tile(tf.expand_dims(T_single, axis=-1),[1,1,40])

        GAT_layer_T_encode = GraphAttention(num_feature=40,name='{}'.format('T_encode'),heads=4)
        # T_add_env = Dense(self.feature_num)(tf.concat((Temperature_his_data,T_env_his),axis=-1))
        T_x,_ = GAT_layer_T_encode(Temperature_his_data-T_env_his,self.liquid_adj)
        T_x = LSTM(units=self.feature_num,input_shape=(self.time_read,self.feature_num),
                            return_sequences=True)(T_x)
        T_code = Conv1D(filters=1,kernel_size=1,activation='relu',
                            input_shape=(self.feature_num,self.feature_num))(T_x)

        ### Humidity_encode model
        Humidity_his_data = Input(shape=(self.time_read, self.feature_num))
        h_id = int(self.dict['humidity'])
        H_env_his = env_his_data[:,:,h_id]
        H_env_his = tf.tile(tf.expand_dims(H_env_his, axis=-1),[1,1,40])
        H_single = env_data[:,:,h_id]
        H_single = tf.tile(tf.expand_dims(H_single, axis=-1),[1,1,40])

        GAT_layer_H_encode = GraphAttention(num_feature=40,name='{}'.format('H_encode'),heads=4)
        # H_add_env = Dense(self.feature_num)(tf.concat((H_env_his,Humidity_his_data),axis=-1))
        H_x,_ = GAT_layer_H_encode(H_env_his-Humidity_his_data,self.ca_adj)
        H_x = LSTM(units=self.feature_num,input_shape=(self.time_read,self.feature_num),
                            return_sequences=True)(H_x)
        H_code = Conv1D(filters=1,kernel_size=1,activation='relu',
                            input_shape=(self.feature_num,self.feature_num))(H_x)
        ### HFR_encode model
        hfr_his_data = Input(shape=(self.time_read, self.feature_num))
        GAT_layer_Hfr_encode = GraphAttention(num_feature=40,name='{}'.format('Hfr_encode'),heads=4)
        hfr_x,_ = GAT_layer_Hfr_encode(hfr_his_data,self.ca_adj)
        hfr_x =LSTM(units=self.feature_num,input_shape=(self.time_read,self.feature_num),
                            return_sequences=True)(hfr_x)
        hfr_code = Conv1D(filters=1,kernel_size=1,activation='relu',
                            input_shape=(self.feature_num,self.feature_num))(hfr_x)
        ### density
        density_predict = Input(shape=(self.time_predict, self.feature_num))
        density_code = Dense(self.feature_num)(density_predict)
        density_code = tf.transpose(density_code,perm=[0,2,1])
        code = tf.concat([T_code,H_code,hfr_code,density_code],-1)

        ###Temperature_Decoder
        T_decode = Dense(self.feature_num*2)(code)
        T_decode = Dense(self.feature_num)(T_decode)
        t_gate_layer = MultiHead_Attention(head_num=4,activation=None,name='t_gate_layer')
        T_gate =tf.nn.sigmoid(t_gate_layer(T_decode))
        t = LSTM(units=self.feature_num,input_shape=(self.time_read,self.feature_num),
                            return_sequences=True)(T_decode*T_gate)
        GAT_layer_T_decode = GraphAttention(num_feature=40,name='{}'.format('T_decode'),heads=4)
        t,_ = GAT_layer_T_decode(t,self.liquid_adj)
        t = tf.transpose(t,perm=[0,2,1])
        T_predict_result =Conv1D(filters=1,kernel_size=1,activation='tanh',
                            input_shape=(self.feature_num,self.feature_num))(t)
        T_predict_result = tf.transpose(T_predict_result,perm=[0,2,1])
        # T_result = Dense(self.feature_num,name='Temperature')(tf.concat((T_predict_result,T_single),axis=-1))
        T_result=Lambda(lambda x:x[0]+x[1],name='Temperature')([T_predict_result,T_single])

        ### Humidity_Decoder
        H_decode = Dense(self.feature_num*2)(code)
        H_decode = Dense(self.feature_num)(H_decode)
        H_gate_layer=MultiHead_Attention(head_num=4,activation=None,name='H_gate_layer')
        H_gate =tf.nn.sigmoid(H_gate_layer(H_decode))
        h = LSTM(units=self.feature_num,input_shape=(self.time_read,self.feature_num),
                            return_sequences=True)(H_decode*H_gate)
        GAT_layer_H_decode = GraphAttention(num_feature=40,name='{}'.format('H_decode'),heads=4)
        h,_ = GAT_layer_H_decode(h,self.ca_adj)
        h = tf.transpose(h,perm=[0,2,1])
        H_predict_result =Conv1D(filters=1,kernel_size=1,activation='tanh',
                            input_shape=(self.feature_num,self.feature_num))(h)
        H_predict_result = tf.transpose(H_predict_result,perm=[0,2,1])
        # H_result=Dense(self.feature_num,name='Humidity')(tf.concat((H_predict_result,H_single),axis=-1))
        H_result=Lambda(lambda x:x[0]+x[1],name='Humidity')([H_predict_result,H_single])

        ### hfr_Decoder
        hfr_decode = Dense(self.feature_num*2)(code)
        hfr_decode = Dense(self.feature_num)(hfr_decode)
        hfr_gate_layer=MultiHead_Attention(head_num=4,activation=None,name='hfr_gate_layer')
        hfr_gate =tf.nn.sigmoid(hfr_gate_layer(hfr_decode))
        hfr =LSTM(units=self.feature_num,input_shape=(self.time_read,self.feature_num),
                            return_sequences=True)(hfr_decode*hfr_gate)
        GAT_layer_Hfr_decode = GraphAttention(num_feature=40,name='{}'.format('Hfr_decode'),heads=4)
        hfr,_ = GAT_layer_Hfr_decode(hfr,self.ca_adj)
        hfr = tf.transpose(hfr,perm=[0,2,1])
        hfr_predict_result =Conv1D(filters=1,kernel_size=1,activation='sigmoid',
                            input_shape=(self.feature_num,self.feature_num))(hfr)
        hfr_result = tf.transpose(hfr_predict_result,perm=[0,2,1])
        hfr_result =Lambda(lambda x:x[0]+(x[1]-x[1]),name='Hfr')([hfr_result,T_single])
        return Model(inputs=[Temperature_his_data,Humidity_his_data,hfr_his_data,density_predict,env_his_data,env_data],
                     outputs=[T_result,H_result,hfr_result])

    def plot_history(self,history,name):
        mape = history.history['{}_mape'.format(name)]
        val_mape = history.history['val_{}_mape'.format(name)]
        loss = history.history['{}_loss'.format(name)]
        val_loss = history.history['val_{}_loss'.format(name)]

        fig = plt.figure(figsize=(11,11))
        plt.plot(100-np.array(mape), label="Train_Acc")
        plt.plot(100-np.array(val_mape), label="Test_Acc")
        font = {'family':'Times New Roman','weight':'normal','size':30}
        plt.tick_params(labelsize=20)
        plt.xlabel('Time',font)
        plt.ylabel('Accuracy',font)
        plt.legend(loc='best',prop={'size':20})
        plt.savefig('./result/{}__Model{}_{}_{}_Acc.png'.format(self.mode, self.version, name, self.epoch))

        fig = plt.figure(figsize=(11, 11))
        plt.plot(loss, label="Train_Loss")
        plt.plot(val_loss, label="Test_Loss")
        font = {'family':'Times New Roman','weight':'normal','size':30}
        plt.tick_params(labelsize=20)
        plt.xlabel('time',font)
        plt.ylabel('Mean Square Error',font)
        plt.legend(loc='best',prop={'size':20})
        plt.savefig('./result/{}__Model{}_{}_{}_MSE.png'.format(self.mode, self.version, name, self.epoch))

if __name__ =='__main__':
    PEMFC = PEMFC_model()