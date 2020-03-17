import pandas as pd
from datetime import datetime,timedelta

df=pd.read_excel(r'C:\Users\GS\Desktop\瓦房电500KV\油色谱数据副本（22M00000022375815）.xlsx',
                 encoding="gbk",)
#将df中的RESAVE_TIME列转换为字符串形式，进行拆分
df["RESAVE_TIME"] = df.RESAVE_TIME.astype(str)
#将df中的RESAVE_TIME列以空格拆分成多列（两列）
df = pd.concat([df,df['RESAVE_TIME'].str.split(' ',expand=True)],axis=1).reset_index(drop=True)
df.rename(columns={0:'date',1:'time'}, inplace=True)
df.drop(columns=['RESAVE_TIME'],inplace=True)
#以data列作为索引
df.set_index('date',inplace=True)
df.index = pd.to_datetime(df.index)  # 将字符串索引转换成时间索引

#取出含有气体的列
colums = ['H2','CO','CO2','CH4','C2H4','C2H2','TOTALHYDROCARBON']
data = df[colums].sort_index(ascending=True)
data = data.groupby(data.index).mean()
data = data['H2']
data.to_excel('error_data.xlsx')
print(data)