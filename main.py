# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 17:35:58 2021

@author: 82109
"""
import seaborn as sn
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
#데이터 전처리
###################################################################################################################
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
merge_holiday = pd.merge(hol,hol_effet)
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
total = pd.merge(total,temp,how="left", on =["year",'month','day','weekday','시간'])
#total에 humidity 추가
avg_humidity = pd.read_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/avg_humidity.csv",encoding= "UTF-8")
#연월일에서 column 추출
avg_humidity['time'] = pd.to_datetime(avg_humidity['time'])
avg_humidity['year'] = avg_humidity['time'].dt.year
avg_humidity['month'] = avg_humidity["time"].dt.month
avg_humidity['day'] = avg_humidity['time'].dt.day
avg_humidity['weekday'] = avg_humidity['time'].dt.weekday
avg_humidity['시간'] = avg_humidity['time'].dt.time
avg_humidity['시간'] = avg_humidity['시간'].apply(lambda x: x.strftime('%H')).astype(int) #datetime에서 시간만빼서 int로 저장
avg_humidity = avg_humidity.drop(['time'],axis=1) #연월일 빼기
total = pd.merge(total,avg_humidity,how="left", on =["year",'month','day','weekday','시간'])


###########################################################################################################
#상관관계 그래프
total = total.rename(columns={'시간':'time',"구분":"kind_of_gas","추정치":"holiday_estimate","공급량":"suply"})
test = test.rename(columns={'시간':'time',"구분":"kind_of_gas","추정치":"holiday_estimate","공급량":"suply"})
test= test_humidity.drop(["Unnamed: 0","연월일"],axis =1)

plt.figure(figsize=(12,8))
sn.heatmap(test.corr(), annot=True)
plt.title("Correlation")
#학습/ 테스트 데이터 설정
###################################################################################################################################

#학습데이터 설정
total = total.rename(columns={'시간':'time',"구분":"kind_of_gas","추정치":"holiday_estimate","공급량":"suply"})
y_train = total[["year","month","day","time","weekday","kind_of_gas","suply"]]
X_train = total.drop(["year","연월일","suply"],axis = 1)

#테스트데이터 (3월 31일까지 데이터)
X_test = pd.DataFrame()
X_test["연월일"] = pd.Series(pd.date_range("1/1/2019",freq= "h",periods = 2160))
X_test["year"] = X_test["연월일"].dt.year
X_test["month"] = X_test["연월일"].dt.month
X_test['day'] = X_test['연월일'].dt.day
X_test['weekday'] = X_test['연월일'].dt.weekday
X_test['시간'] = X_test['연월일'].dt.time
X_test["시간"] = X_test['시간'].apply(lambda x: x.strftime('%H')).astype(int)
#특수일 추가
X_test["특수일"] = X_test["weekday"].apply(lambda x : "근무일" if x  in [0,1,2,3,4] else "주말")
X_test= pd.merge(X_test,merge_holiday, on = ["year",'month','day','weekday'],how = "left")
X_test["특수일_y"] = np.where(pd.notnull(X_test["특수일_y"]) == True, X_test['특수일_y'], X_test['특수일_x']) # 조건문으로 nan값 채우기
predict_rate = []
for idx, val in enumerate(X_test["특수일_y"]):
    if val == "근무일":
        predict_rate.append(1)
    elif val == "주말":
        predict_rate.append(-0.165)
    else:
        predict_rate.append(X_test["추정치"][idx])
X_test["추정치"] = predict_rate
X_test = X_test.drop(["year","연월일_x","연월일_y"],axis = 1)
#구분별로 똑같은거 7개 만들기
X_test = pd.concat([X_test,X_test,X_test,X_test,X_test,X_test,X_test],axis=0)
X_test
#구분 넣어주기
b0= pd.Series([0]*2160)
b1 = pd.Series([1]*2160)
b2 = pd.Series([2]*2160)
b3 = pd.Series([3]*2160)
b4 = pd.Series([4]*2160)
b5 = pd.Series([5]*2160)
b6 = pd.Series([6]*2160)
a= pd.DataFrame()
a["구분"] = pd.concat([b0,b1,b2,b3,b4,b5,b6],axis=0)
X_test["구분"] = a["구분"]
#기온과 습도 2019년 데이터 쓰기
test_temp = pd.read_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/test_2019.csv",encoding = "UTF-8")
test_temp
X_test[["temp","humidity"]] = pd.concat([test_temp[["temp","humidity"]][:2160],test_temp[["temp","humidity"]][:2160],test_temp[["temp","humidity"]][:2160],test_temp[["temp","humidity"]][:2160],test_temp[["temp","humidity"]][:2160],test_temp[["temp","humidity"]][:2160],test_temp[["temp","humidity"]][:2160]],axis=0)
X_test = X_test.rename(columns={'시간':'time',"구분":"kind_of_gas","추정치":"holiday_estimate","공급량":"suply"})
#X_test.to_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/X_test.csv",index=False)
X_test[2160:]

################################################################################################   
#구분별로 time, month, hoiday_estimate, temp 만 사용
index = ["day","month","temp","humidity","holiday_estimate"]

#1월 ~ 12월31일까지 구분별 학습데이터

#학습 데이터 구분별 52584개
X_train_A = X_train[X_train["kind_of_gas"]==0][index]
y_train_A = y_train[(y_train["kind_of_gas"]==0)]["suply"]

X_train_B = X_train[(X_train["kind_of_gas"]== 1)][index]
y_train_B = y_train[(y_train["kind_of_gas"]==1)]["suply"]

X_train_C = X_train[(X_train["kind_of_gas"]== 2)][index]
y_train_C = y_train[(y_train["kind_of_gas"]==2)]["suply"]

X_train_D = X_train[(X_train["kind_of_gas"]== 3)][index]
y_train_D = y_train[(y_train["kind_of_gas"]==3)]["suply"]

X_train_E = X_train[(X_train["kind_of_gas"]== 4)][index]
y_train_E = y_train[(y_train["kind_of_gas"]==4)]["suply"]

X_train_F = X_train[(X_train["kind_of_gas"]== 5)][index]
y_train_F = y_train[(y_train["kind_of_gas"]==5)]["suply"]
                    
X_train_G = X_train[(X_train["kind_of_gas"]== 6)][index]
y_train_G = y_train[(y_train["kind_of_gas"]==6)]["suply"]

#1월~12월31일 까지 구분별 테스트데이터
X_test_A = X_test[X_test["구분"]==0][index]
X_test_B = X_test[X_test["구분"]==1][index]
X_test_C = X_test[X_test["구분"]==2][index]
X_test_D = X_test[X_test["구분"]==3][index]
X_test_E = X_test[X_test["구분"]==4][index]
X_test_F = X_test[X_test["구분"]==5][index]
X_test_G = X_test[X_test["구분"]==6][index]

#1월 ~ 3월31일까지
X_test_A = X_test[(X_test["kind_of_gas"]==0)&(X_test["month"]<4)][index]
X_test_B = X_test[(X_test["kind_of_gas"]==0)&(X_test["month"]<4)][index]
X_test_C = X_test[(X_test["kind_of_gas"]==0)&(X_test["month"]<4)][index]
X_test_D = X_test[(X_test["kind_of_gas"]==0)&(X_test["month"]<4)][index]
X_test_E = X_test[(X_test["kind_of_gas"]==0)&(X_test["month"]<4)][index]
X_test_F = X_test[(X_test["kind_of_gas"]==0)&(X_test["month"]<4)][index]
X_test_G = X_test[(X_test["kind_of_gas"]==0)&(X_test["month"]<4)][index]
#모델선택
#########################################################################################################
#XGB
from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train_A,y_train_A)
pred_xgb_A = xgb.predict(X_test_A)

xgb.fit(X_train_B,y_train_B)
pred_xgb_B = xgb.predict(X_test_B)

xgb.fit(X_train_C,y_train_C)
pred_xgb_C = xgb.predict(X_test_C)

xgb.fit(X_train_D,y_train_D)
pred_xgb_D = xgb.predict(X_test_D)

xgb.fit(X_train_E,y_train_E)
pred_xgb_E = xgb.predict(X_test_E)

xgb.fit(X_train_F,y_train_F)
pred_xgb_F = xgb.predict(X_test_F)

xgb.fit(X_train_G,y_train_G)
pred_xgb_G = xgb.predict(X_test_G)

#그리드 서치
from sklearn import model_selection
xgb_parameters ={'max_depth' : [3,4,5,6] , 'n_estimators': [12,24,32], 'learning_rate':[0.01, 0.1], 'gamma': [0.5, 1, 2], 'random_state':[99]}
grid_search_xgb = model_selection.GridSearchCV ( estimator = xgb, param_grid = xgb_parameters, scoring = 'recall', cv = 10 )
grid_search_xgb.fit(X_train_A,y_train_A )
best_xgb_parameter = grid_search_xgb.best_estimator_
best_xgb_parameter

#결과값 합치기
pred_xgb = np.concatenate([pred_xgb_A,pred_xgb_B,pred_xgb_C,pred_xgb_D,pred_xgb_E,pred_xgb_F,pred_xgb_G],axis = 0)


#제출
submission = pd.read_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/sample_submission.csv",encoding="UTF-8")
submission["공급량"] = pred_xgb
submission.to_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/submission(그리드서치).csv",encoding="UTF-8",index=False)
pred_xgb1 = pd.DataFrame()
pred_xgb1["pred"] = pred_xgb
pred_xgb1.to_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/pred_xgb(2019년)).csv",encoding="UTF-8",index=False)
#################################################################################################################
#평가


