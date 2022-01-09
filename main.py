# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 17:35:58 2021

@author: 82109
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
plt.style.use("ggplot")
warnings.filterwarnings('ignore')

total = pd.read_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/한국가스공사_시간별 공급량_20181231.csv", encoding="CP949")
hol = pd.read_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/한국가스공사_대전 도시가스 수요의 특수일 효과_20210914.csv", encoding="CP949")

#d_map 딕셔너리를 생성해, 구분 encoding!      ex) A -> 0, B -> 1, ...
d_map = {}
for i, d in enumerate(total['구분'].unique()):
    d_map[d] = i
total['구분'] = total['구분'].map(d_map)
total['연월일'] = pd.to_datetime(total['연월일'])
#연월일에서 column 추출
total['year'] = total['연월일'].dt.year
total['month'] = total['연월일'].dt.month
total['day'] = total['연월일'].dt.day
total['weekday'] = total['연월일'].dt.weekday

train_years = [2013,2014,2015,2016,2017]
val_years = [2018]

train = total[total['year'].isin(train_years)]
val = total[total['year'].isin(val_years)]
total['공급량'][0]*(1+hol['추정치'][18])
df= pd.read_csv("C:/Users/user/Documents/GitHub/Dacon_NLP_Technology_Classification/data/train.csv", encoding="UTF-8")
