#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:07:54 2020

@author: jennychou
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


class Models:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self._models = []
        self._scores = []
        self._mse = []
        self._best_model = None
        self._best_mse = None
    
    def set_baseline(self, baseline):
        self._mse.append(baseline)

    def add_model(self, model):
        self._models.append(model)

    def cv_models(self, cv=None):
        for model in self._models:
            scores = cross_val_score(model, self.x, self.y, cv=cv,
                                     scoring='neg_mean_squared_error')
            mse = -1.0 * np.mean(scores)
            self._print_summary(model, scores, mse)

            if(mse <= min(self._mse)):
                self._best_model = model
                self._best_mse = mse
                
            self._scores.append(scores)
            self._mse.append(mse)

    def print_all(self):
        print("Models:", self._models)
        print("Scores:", self._scores)
        print("MSE:", self._mse)
        print()
        
    def print_best_model(self):
        print("Best model: %s\nBest MSE: %.3f" % 
              (self._best_model, self._best_mse))
        self._best_model.fit(self.x, self.y)
        
        if hasattr(self._best_model, 'feature_importances_'):
            feature_importance = self._best_model.feature_importances_
            index = self.x.columns
            for i in range(len(feature_importance)):
                print("%s  %.6f" % (index[i], feature_importance[i]))
        print()
                
    def predict(self, test):
        test_pred = self._best_model.predict(test).round(3)
        return pd.DataFrame(test_pred, index=test.index)
    
    def save_to_csv(self, test_pred, filename):
        test_pred.to_csv(filename)
        
    def _print_summary(self, model, scores, mse):
        print("Model:", model)
        print("Scores:", *scores)
        print("MSE: %.3f" % mse)
        print()


dataset = FeatureEngineer(os.path.join("data", "train_features.csv"),
                          os.path.join("data", "train_salaries.csv"),
                          os.path.join("data", "test_features.csv"),
                          'salary',
                          'jobId')
features = ['companyId', 'jobType', 'degree', 'major', 'industry']
dataset.add_stats(features, "CJDMI")

dataset.encode_data()

x, y = dataset.get_x_y()
model = Models(x, y)
model.set_baseline(dataset.get_baseline())

model.add_model(LinearRegression())
model.add_model(Lasso())
model.add_model(GradientBoostingRegressor(max_depth=8))
model.cv_models()

model.print_all()
model.print_best_model()

filename = os.path.join("data", "test_salaries_prediction.csv")
model.save_to_csv(model.predict(dataset.test), filename)

