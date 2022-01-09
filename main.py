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

###################################################################################################################
#데이터 전처리
#명목형변수 처리, 구분 encoding!      ex) A -> 0, B -> 1, ...
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
#############################################################################################################
#total에 특수일가중치 추가
hol_effet = pd.read_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/한국가스공사_대전 도시가스 수요의 특수일 효과_수정.csv", encoding="CP949")
hol = pd.read_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/2013~2018공휴일_수정.csv",index_col=0,encoding="CP949")

#연월일에서 column 추출
hol['연월일'] = pd.to_datetime(hol['연월일'])
hol['year'] = hol['연월일'].dt.year
hol['month'] = hol['연월일'].dt.month
hol['day'] = hol['연월일'].dt.day
hol['weekday'] = hol['연월일'].dt.weekday

#total에 근무일 주말 추가 
total["특수일"] = total["weekday"].apply(lambda x : "근무일" if x  in [0,1,2,3,4] else "주말")
total = pd.merge(total,merge_holiday, on = ["연월일","year",'month','day','weekday'],how = "left")
total["특수일_y"] = np.where(pd.notnull(total["특수일_y"]) == True, total['특수일_y'], total['특수일_x']) # 조건문으로 nan값 채우기
predict_rate = []
for idx, val in enumerate(total["특수일_y"]):
    if val == "근무일":
        predict_rate.append(1)
    elif val == "주말":
        predict_rate.append(-0.165)
    else:
        predict_rate.append(total["추정치"][idx])
total["추정치"] = predict_rate
###############################################################################################################
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
#merge_temp = pd.merge(total,temp,how="left", on =["year",'month','day','weekday','시간'])
total = pd.merge(total,temp,how="left", on =["year",'month','day','weekday','시간'])
###########################################################################################################


#2017년 10월3일 개천절+추석이라 24개 데이터 늘어남 껄껄
#추가적으로 평일과 주말 나눠서 추정치 넣어줘야됨 #해결

#데이터 저장
total = total.drop(["Unnamed: 0","Unnamed: 0.1"],axis = 1)
total.to_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/merge_total_211206.csv",encoding="CP949",index=False)
###################################################################################################################################





total = pd.read_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/merge_total_211206.csv",encoding="CP949")
submission = pd.read_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/sample_submission.csv",encoding="UTF-8")

y_train = total[["year","month","day","시간","weekday","구분","공급량"]]
X_train = total.drop(["연월일","공급량"],axis = 1)
#명목형데이터 처리
X_train = pd.get_dummies(X_train)
#수치형데이터 처리
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#데이터 나누기
#from sklearn.model_selection import train_test_split
X_valid1,X_valid2,y_valid1,y_valid2 = train_test_split(X_train,y_train, test_size = 0.1)
print(15120*6)
X_valid = X_train[(X_train["month"]<4)].drop("year",axis =1)
y_valid = y_train[(y_train["month"]<4)]

X_test = X_valid[:15120]
X_train = X_train.drop("year",axis =1)
#명목형데이터 처리


X_train_A = X_train[(X_train["구분"]== 0)&(X_train["month"]<4)].drop("구분",axis =1)
y_train_A = y_train[(y_train["구분"]==0)&(y_train["month"]<4)]

X_train_B = X_train[(X_train["구분"]== 1)&(X_train["month"]<4)].drop("구분",axis =1)
y_train_B = y_train[(y_train["구분"]==1)&(y_train["month"]<4)]

X_train_C = X_train[(X_train["구분"]== 2)&(X_train["month"]<4)].drop("구분",axis =1)
y_train_C = y_train[(y_train["구분"]==2)&(y_train["month"]<4)]

X_train_D = X_train[(X_train["구분"]== 3)&(X_train["month"]<4)].drop("구분",axis =1)
y_train_D = y_train[(y_train["구분"]==3)&(y_train["month"]<4)]

X_train_E = X_train[(X_train["구분"]== 4)&(X_train["month"]<4)].drop("구분",axis =1)
y_train_E = y_train[(y_train["구분"]==4)&(y_train["month"]<4)]

X_train_F = X_train[(X_train["구분"]== 5)&(X_train["month"]<4)].drop("구분",axis =1)
y_train_F = y_train[(y_train["구분"]==5)&(y_train["month"]<4)]
                    
X_train_G = X_train[(X_train["구분"]== 6)&(X_train["month"]<4)].drop("구분",axis =1)
y_train_G = y_train[(y_train["구분"]==6)&(y_train["month"]<4)]

#모델선택
from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train,y_train["공급량"])
pred_xgb = xgb.predict(X_test)

submission["공급량"] = pred_xgb
submission.to_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/submission_211206.csv",encoding="CP949",index=False)

#평가

from sklearn.metrics import r2_score
r2_score(y_train["공급량"][:15120],pred_xgb)

np.mean((np.abs(y_train["공급량"][:15120]-pred_xgb))/y_train["공급량"][:15120])

#test에 공급량에 대한 가중치를 알려면 2019년의 공휴일과 평일 주말을 대입시켜야