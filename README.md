2021년(제2회) NH투자증권 빅데이터 경진대회
데이터 정의
1.고객 및 주거래계좌 정보 (CUS_INFO.CSV)

act_id: 계좌 ID

sex_dit_cd: 성별

cus_age_stn_cd: 연령대

ivs_icn_cd: 투자성향

cus_aet_stn_cd: 자산구간

mrz_pdt_tp_sgm_cd: 주거래상품군

lsg_sgm_cd: Life Style

tco_cus_grd_cd: 서비스 등급

tot_ivs_te_sgm_cd: 총 투자기간

mrz_btp_dit_cd: 주거래업종구분

2.국내주식 잔고이력 (STK_BNC_HIST.CSV) act_id: 계좌 ID

bse_dt: 기준일자

iem_cd: 종목코드

bnc_qty: 잔고수량

tot_aet_amt: 잔고금액

stk_par_pr: 주당 액면가

3.국내주식 보유기간(STK_HLD.CSV)

이경우 train,test 로 분리되어있음

stk_hld_train.csv(681,472건): 16년 1월 ~ 20년 12월 사이 고객의 국내주식 거래가 -종료 된 건

stk_hld_test.csv(70,596건): 20년 12월 이전에 매수하고 21년 이후에 고객이 전량 매도한 국내주식 보유기간 예측

act_id: 계좌 ID

iem_cd: 종목코드

byn_dt: 매수일자

hold_d: 보유기간(일)

4.종목정보(IEM_INFO.CSV)

iem_cd: 종목코드

iem_krl_nm: 종목한글명

btp_cfc_cd: 종목업종

mkt_pr_tal_scl_tp_cd: 시가총액 규모유형

stk_dit_cd: 시장구분

분석목적
주식 보유기간 예측 및 서비스 아이디어 제안

고객이 보유한 주식 종목 별 보유기간 예측

외부데이터와 결합하여 보유기간을 예측하는 모델링을 수행 (ex: 코스피 지수 국가 ,코로나발생으로인한 종목별 차이등)

평가방법
RMSE

(자세한 데이터 명세는 https://www.dacon.io/competitions/official/235798/talkboard/404251?page=1&dtype=recent)
