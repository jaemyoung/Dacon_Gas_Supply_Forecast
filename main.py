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
from datetime import datetime, timedelta

#전체데이터 불러오기
total = pd.read_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/한국가스공사_시간별 공급량_20181231.csv", encoding="CP949")

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
total['시간'] =total['시간']-1 #시간단위 맞추기위해 -1해주기


#total에 temp추가
temp = pd.read_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/avg_temp.csv",encoding="CP949")
#연월일에서 column 추출
temp['연월일'] = pd.to_datetime(temp['연월일'])
temp['year'] = temp['연월일'].dt.year
temp['month'] = temp['연월일'].dt.month
temp['day'] = temp['연월일'].dt.day
temp['weekday'] = temp['연월일'].dt.weekday
temp['시간'] = temp['연월일'].dt.time

temp['시간'] = temp['시간'].apply(lambda x: x.strftime('%H')).astype(int) #datetime에서 시간만빼서 int로 저장
temp = temp.drop(['연월일'],axis=1) #연월일 빼기
merge_temp = pd.merge(total,temp,how="left", on =["year",'month','day','weekday','시간'])

#total에 특수일 추가
hol_effet = pd.read_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/한국가스공사_대전 도시가스 수요의 특수일 효과_20210914.csv", encoding="CP949")
hol = pd.read_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/2013~2018공휴일.csv",index_col=0,encoding="CP949")
#연월일에서 column 추출
hol['연월일'] = pd.to_datetime(hol['연월일'])
hol['year'] = hol['연월일'].dt.year
hol['month'] = hol['연월일'].dt.month
hol['day'] = hol['연월일'].dt.day
hol['weekday'] = hol['연월일'].dt.weekday

merge_holiday = pd.merge(hol,hol_effet,how='inner') #특수일 추가
merge_total = pd.merge(merge_temp,merge_holiday,how="left") #total로 merge
#merge_total[merge_total['구분']==0] 확인
#merge_total[merge_total['구분']==1]

#2017년 10월3일 개천절+추석이라 24개 데이터 늘어남 껄껄
#추가적으로 평일과 주말 나눠서 추정치 넣어줘야됨

merge_total.to_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/merge_total_211117.csv",encoding="CP949")

total['year'].apply(lambda x : "2016년 이하" if x <2016 else "2016년초과")
