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
###########################################################################################################

#학습/ 테스트 데이터 설정
###################################################################################################################################
#제출Csv
submission = pd.read_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/sample_submission.csv",encoding="UTF-8")

#학습데이터 설정
y_train = total[["year","month","day","시간","weekday","구분","공급량"]]
X_train = total.drop(["year","연월일","공급량"],axis = 1)
#학습데이터 = 1월 ~3월 31일까지의 데이터
X_valid1 = X_train[(X_train["month"]<4)]
y_valid1 = y_train[(y_train["month"]<4)]
#학습데이터 = 1월~12월31일까지의 데이터
X_valid2 = X_train
y_valid2 = y_train

#테스트데이터 (3월 31일까지 데이터)
X_test = pd.DataFrame()
X_test["연월일"] = pd.Series(pd.date_range("1/1/2019",freq= "h",periods = 2160))
X_test["year"] = X_test["연월일"].dt.year
X_test["month"] = X_test["연월일"].dt.month
X_test['day'] = X_test['연월일'].dt.day
X_test['weekday'] = X_test['연월일'].dt.weekday
X_test['시간'] = X_test['연월일'].dt.time
X_test['시간'] = X_test['시간'].apply(lambda x: x.strftime('%H')).astype(int)
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
#기온은 2013년 데이터 쓰기
test_temp = X_valid[:2160]["temp"]
X_test = pd.concat([X_test,test_temp],axis = 1)
#구분별로 똑같은거 7개 만들기
X_test = pd.concat([X_test,X_test,X_test,X_test,X_test,X_test,X_test],axis=0)
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
X_test.to_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/X_test.csv",index =False)




#스케일러
############################################################
#수치형데이터 처리
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#명목형데이터 처리
concat = pd.concat([X_valid,X_test])
concat = pd.get_dummies(concat)
X_vaild = concat[:90888]
X_test = concat[90888:]
###################################################################    
#구분별학습

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
xgb.fit(X_vaild,y_valid["공급량"])
pred_xgb = xgb.predict(X_test)

submission["공급량"] = pred_xgb
submission.to_csv("C:/Users/user/Documents/GitHub/Dacon_Gas_Supply_Forecast/Data/submission_211206(요일추가).csv",encoding="UTF-8",index=False)

#평가

from sklearn.metrics import r2_score
r2_score(y_train["공급량"][:15120],pred_xgb)

np.mean((np.abs(y_train["공급량"][:15120]-pred_xgb))/y_train["공급량"][:15120])

#test에 공급량에 대한 가중치를 알려면 2019년의 기온을 대입시켜야