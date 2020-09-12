# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 09:55:00 2020

@author: jenny
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
pd.set_option('precision', 3)


class Data:
    def __init__(self, train_file, target_file, test_file, target, index):
        self.train = None
        self.test = None
        self.features = None
        self.target = target
        self.index = index

        self._load_data(train_file, target_file, test_file)
        self._clean_data()
        print("After clean up train dataset has shape:", self.train.shape)
        print("After clean up test dataset has shape:", self.test.shape)
        print()

    def encode_data(self):
        for col in self.features:
            group_dict = dict(self.train.groupby([col])[self.target].mean())
            self.train[col] = self.train[col].map(group_dict)
            self.test[col] = self.test[col].map(group_dict)

    def get_x_y(self):
        return self.train.drop(columns=self.target), self.train[self.target]
    
    def get_baseline(self):
        baseline_true = self.train[self.target].values.astype(float)
        mean_dict = dict(self.train.groupby(['industry'])[self.target].mean())
        baseline_pred = self.train.industry.map(mean_dict)
        baseline_mse = mean_squared_error(baseline_true, baseline_pred)
        print("Baseline: MSE=%.3f\n" % baseline_mse)
        return baseline_mse

    def _load_data(self, train_file, target_file, test_file):
        self.train = pd.read_csv(train_file)
        self.features = self.train.drop(columns=self.index).columns.values
        self.train = pd.merge(self.train, pd.read_csv(target_file), on=self.index)
        self.test = pd.read_csv(test_file)

    def _clean_data(self):
        self._drop_duplicates(self.train)
        self._drop_null(self.train)
        self._check_col_validity(self.train, 'yearsExperience', 0)
        self._check_col_validity(self.train, 'milesFromMetropolis', 0)
        self._check_col_validity(self.train, 'salary', 1)

    def _drop_duplicates(self, data):
        print("Remove %d duplicated jobs" % data.duplicated().sum())
        data.drop_duplicates(inplace=True)

    def _drop_null(self, data):
        invalid_jobs = data.index[data.isnull().sum(axis=1).gt(0)].values
        print("Remove %d jobs with missing values" % len(invalid_jobs))
        data.drop(index=invalid_jobs, inplace=True)

    def _check_col_validity(self, data, col, lt):
        invalid_jobs = data.index[data[col].lt(lt)]
        print("Remove %d jobs with invalid %s" % (len(invalid_jobs), col))
        data.drop(index=invalid_jobs, inplace=True)


class FeatureEngineer(Data):
    def __init__(self, train_file, target_file, test_file, target, index):
        Data.__init__(self, train_file, target_file, test_file, target, index)
        self._stats = []

    def add_stats(self, cols, col_name):
        self._generate_stats(cols, col_name)
        self._add_stats(cols, col_name)
        self.train.set_index(self.index, inplace=True)
        self.test.set_index(self.index, inplace=True)

    def _generate_stats(self, cols, col_name):
        group = self.train.groupby(cols)[self.target]
        Q1 = group.quantile(0.25)
        Q3 = group.quantile(0.75)
        upper_bound = Q3 + 1.5 * (Q3 - Q1)
        self._stats = pd.DataFrame({col_name+"_mean" : group.mean()})
        self._stats[col_name + "_min"] = group.min()
        self._stats[col_name + "_Q1"] = Q1
        self._stats[col_name + "_median"] = group.median()
        self._stats[col_name + "_Q3"] = Q3
        self._stats[col_name + "_upper"] = upper_bound
        self._stats[col_name + "_max"] = group.max()

    def _add_stats(self, cols, col_name):
        self._generate_stats(cols, col_name)
        self.train = pd.merge(self.train, self._stats, on=cols)
        self.test = pd.merge(self.test, self._stats, on=cols)


dataset = FeatureEngineer(os.path.join("data", "train_features.csv"),
                          os.path.join("data", "train_salaries.csv"),
                          os.path.join("data", "test_features.csv"),
                          'salary',
                          'jobId')
features = ['companyId', 'jobType', 'degree', 'major', 'industry']
dataset.add_stats(features, "CJDMI")

dataset.encode_data()

x, y = dataset.get_x_y()

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(x.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1)
    ])
model.compile(loss='mse', optimizer=tf.optimizers.Adam(), metrics=['mse'])
weightpath = os.path.join("dnn_outputs", 
                          "best-weight-batch_size_1000-epochs_100.hdf5")
checkpoint = tf.keras.callbacks.ModelCheckpoint(weightpath, 
                                                monitor='val_loss', 
                                                verbose=True, 
                                                save_best_only=True, 
                                                mode='auto')
history = model.fit(x, y, batch_size=1000, epochs=100, validation_split=0.1, 
                    callbacks=[checkpoint], shuffle=True)

pngpath = os.path.join("dnn_outputs", "loss-batch_size_1000-epochs_100.png")
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.savefig(pngpath)

csvpath = os.path.join("dnn_outputs", "test_salaries_prediction_dnn.csv")
test_pred = model.predict(dataset.test)
test_final = pd.DataFrame(test_pred, index=dataset.test.index,
                          columns=[dataset.target])
test_final.to_csv(csvpath)

"""
Train on 899995 samples, validate on 100000 samples
Epoch 398/400
881000/899995 [============================>.] - ETA: 0s 
- loss: 299.2771 - mse: 299.2771
Epoch 00398: val_loss did not improve from 313.30743
899995/899995 [==============================] - 2s 3us/sample 
- loss: 299.3032 - mse: 299.3032 - val_loss: 317.0105 - val_mse: 317.0105
Epoch 399/400
899000/899995 [============================>.] - ETA: 0s 
- loss: 299.4860 - mse: 299.4858
Epoch 00399: val_loss did not improve from 313.30743
899995/899995 [==============================] - 3s 3us/sample 
- loss: 299.4861 - mse: 299.4859 - val_loss: 314.1380 - val_mse: 314.1379
Epoch 400/400
886000/899995 [============================>.] - ETA: 0s 
- loss: 299.3097 - mse: 299.3095
Epoch 00400: val_loss did not improve from 313.30743
899995/899995 [==============================] - 2s 3us/sample 
- loss: 299.3113 - mse: 299.3111 - val_loss: 315.4294 - val_mse: 315.4293
"""

