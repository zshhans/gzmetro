import pandas as pd
import numpy as np
from einops import rearrange
import warnings

warnings.filterwarnings('ignore')

# 读取训练数据，改改名字和格式
train_df = pd.read_csv('data/客流量数据.csv',
                       header=0,
                       names=['date', 'time', 'station', 'inflow', 'outflow'],
                       dtype={
                           "date": str,
                           "time": str,
                           "station": str
                       })
date = train_df.date.apply(int)
train_df.station = train_df.station.apply(lambda x: x[0])
train_df['date'] = train_df.apply(lambda x: pd.to_datetime(
    f"{x.date[:4]}-{x.date[4:6]}-{x.date[6:]} {x.time.split('-')[0]}"),
                                  axis=1)
train_df = train_df[['date', 'station', 'inflow', 'outflow']]

# 提取2023年1月30日（春节假期）之后的数据
df1 = train_df[(date >= 20230130)].copy()
df1 = df1.pivot(index='date', columns='station', values=['inflow', 'outflow'])
df1 = df1.fillna(0)
df1.columns = df1.columns.map('_'.join).to_series()
new_date_range = pd.date_range(start="2023-01-30 00:00:00",
                               end="2023-03-31 23:59:59",
                               freq="15min")
df1 = df1.reindex(new_date_range, fill_value=0)
df1 = df1.reset_index()

# 提取2022年2月10日（春节假期）到2022年9月25日（疫情影响严重）的数据
df2 = train_df[(date > 20220210) & (date <= 20220925)].copy()
df2 = df2.pivot(index='date', columns='station', values=['inflow', 'outflow'])
df2 = df2.fillna(0)
df2.columns = df2.columns.map('_'.join).to_series()
new_date_range = pd.date_range(start="2022-02-11 00:00:00",
                               end="2022-09-25 23:59:59",
                               freq="15min")
df2 = df2.reindex(new_date_range, fill_value=0)
df2 = df2.reset_index()

# 合并，格式化
train_df = pd.concat([df2, df1], axis=0)
train_df = train_df.rename(columns={"index": "date"})
train_df.iloc[train_df.date.dt.hour < 6, 1:] = 0
train_df.iloc[:, 1:].apply(lambda x: np.round(x))

# 读取测试数据
test_dfs = []
for station in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    test_dfs.append(
        pd.read_csv(f'data/A-G三天客流数据/{station}.csv',
                    header=0,
                    names=["date", "inflow", "outflow", "station"],
                    parse_dates=[0]))

test_df = pd.concat(test_dfs, axis=0)
test_df = test_df[test_df.date.dt.dayofweek == 4]
test_df = test_df.pivot(index='date',
                        columns='station',
                        values=['inflow', 'outflow'])
test_df.columns = test_df.columns.map('_'.join).to_series()
test_df = test_df.reset_index()
test_df = test_df.fillna(0)
test_df.iloc[test_df.date.dt.hour < 6, 1:] = 0

# 周五去除节假日
fri = train_df[(train_df.date.dt.dayofweek == 4)
               & (train_df.date.dt.date != pd.Timestamp(2022, 4, 29))
               & (train_df.date.dt.date != pd.Timestamp(2022, 6, 3))
               & (train_df.date.dt.date != pd.Timestamp(2022, 9, 9))
               & (train_df.date.dt.date != pd.Timestamp(
                   2023, 3, 31))].iloc[:, 1:].values
# 周六去除节假日
sat = train_df[(train_df.date.dt.dayofweek == 5)
               & (train_df.date.dt.date != pd.Timestamp(2022, 4, 30))
               & (train_df.date.dt.date != pd.Timestamp(2022, 6, 4))
               & (train_df.date.dt.date != pd.Timestamp(2022, 9, 10))
               & (train_df.date.dt.date != pd.Timestamp(2023, 4,
                                                        1))].iloc[:, 1:].values

# station, date, time, channel
fri = rearrange(fri, "(d l) (c s) -> s d l c", l=96, s=7)[:, :, 24:, :]
sat = rearrange(sat, "(d l) (c s) -> s d l c", l=96, s=7)[:, :, 24:, :]

# 按照周五的均值方差标准化
fri_mean = np.mean(fri, axis=2, keepdims=True)
fri_std = np.std(fri, axis=2, keepdims=True)
fri = (fri - fri_mean) / fri_std
sat = (sat - fri_mean) / fri_std

# 用4月7日周五的数据跟训练数据计算相关系数
test = rearrange(test_df.iloc[:, 1:].values,
                    "(d l) (c s) -> s d l c",
                    l=96,
                    s=7)[:, :, 24:, :]
corr = np.zeros((fri.shape[1], 7, 2))
for i in range(fri.shape[1]):
    for s in range(7):
        for c in range(2):
            corr[i, s,
                 c] = (fri[s, i, :, c] * test[s, 0, :, c]).sum() / np.sqrt(
                     (fri[s, i, :, c]**2).sum() * (test[s, 0, :, c]**2).sum())

# 取相关系数最高的N=7天
topk=np.argsort(corr,0)[-7:,:,:]

# 预测
pred=np.zeros((7,1,72,2))
for s in range(7):
    for c in range(2):
        sat_ha=np.mean(sat[s,topk[:,s,c],:,c],axis=0,keepdims=True)
        test_mean=np.mean(test[s,:,:,c],axis=1,keepdims=True)
        test_std=np.std(test[s,:,:,c],axis=1,keepdims=True)
        pred[s,0,:,c]=sat_ha*test_std+test_mean

pred=np.pad(pred,((0,0),(0,0),(24,0),(0,0)),'constant',constant_values=0)
y=rearrange(pred,"s 1 l c -> (s l) c")
y[y < 0] = 0
y = np.round(y).astype(int)

# 输出csv
result_df = pd.read_csv("data/result.csv", header=0)
result_df['Time']=result_df['Time'].apply(lambda x : x.replace("2023/4/16","2023/4/8"))
result_df['InNum'] = y[:, 0]
result_df['OutNum'] = y[:, 1]
result_df.to_csv("result.csv", index=False)
