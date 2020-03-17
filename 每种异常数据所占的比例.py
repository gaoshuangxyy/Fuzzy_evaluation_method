import pandas as pd
import numpy as np
from outliers import smirnov_grubbs as grubbs

data = pd.read_excel('error_data.xlsx',encoding="gbk")

# print(float(data.loc[0][1]))
# print(np.isnan(data.loc[0][1]))
# if np.isnan(data.loc[0][1]):
#     print(222222)

dict_rate = {} #用来存放各个数据异常类型得比率
'''数据长期间中断或短期'''
def X1_X2(data):
    count = 0
    flag1 = 0
    flag2 = 0
    for i in range(len(data)):
        if np.isnan(data.loc[i][1]):
            count = count + 1
        else:
            pass
    if count<7:
        flag1 = 1
        x1_rate = count/len(data)
        dict_rate['X1'] = x1_rate
    else:
        flag2 = 1
        x2_rate = count/len(data)
        dict_rate['X2'] = x2_rate
    if flag1 == 1:
        x2_rate = 0
        dict_rate['X2'] = x2_rate
    if flag2 == 1:
        x1_rate = 0
        dict_rate['X1'] = x1_rate
    return dict_rate

'''数据重复'''
def X3(data):
    data1 = data.dropna()
    # print(data1)
    x3_rate =(len(data) - len(data1.drop_duplicates(subset='H2',keep='first',inplace=False)))/len(data)
    dict_rate['X3'] = x3_rate
    return dict_rate

'''固定偏差（测量值与实际值之间存在固定的偏差）'''
#这个不好从故障数据中统计，我们假设根据专家经验 比例为x4_rate = 0.01
def X4():
    x4_rate = 0.01
    dict_rate['X4'] = x4_rate
    return dict_rate

'''数据为0'''
def X5(data):
    count = 0
    for i in range(len(data)):
        if data.loc[i][1] == 0:
            count += 1
    x5_rate = count/len(data)
    dict_rate['X5'] = x5_rate
    return dict_rate
'''离群点和数据抖动'''
def X8_X9(data):
    # data_copy = data.values[:,1].copy()
    # data_series = pd.Series(value['b'].values, index=value['b'].index)
    temp = set(grubbs.test(data.values[:,1], alpha=0.01))
    # print(temp)
    count = 0
    flag8, flag9 = 0, 0
    # all_right_data = func_jump_error((data_series))
    all_right_data = list(temp)
    for item_right in all_right_data:
        if item_right  not in data.values[:,1]:
            count += 1
    if count < 10:
        flag9 = 1
        dict_rate['X9'] = count/len(data)
    if count >=10:
        flag8 = 1
        dict_rate['X8'] = count/len(data)
    if flag9 == 1:
        dict_rate['X8'] = 0
    if flag8 == 1:
        dict_rate['X9'] = 0
    return dict_rate

def residal(data):#求残差
    sum = 0.0
    mean_ = data.mean()
    # mean_ = np.mean(data)
    for residal_item in data:
        # print(type(float(residal_item)))
        sum+=(float(residal_item)-mean_)**2
    return sum

'''数据跳边'''
def x6_x7(data):
    data = data.values[:,1]
    residal_list,standard_num,standard_num_up = [],0,1
    standard = residal(data)
    residal_list.append(residal(data))
    for item_x6_x7 in range(1,len(data)-1):
        k_temp = residal(data[0:item_x6_x7])+residal(data[item_x6_x7:len(data-1)])
        residal_list.append(k_temp)
        if k_temp>3*standard:
            standard_num+=1
    residal_list.append(residal(data))
    if standard_num_up>1:
        print("有数据突变")
        return 7
    else:
        print("数据连续增长或降低。")
        return 6

def func(x,dict):
    sum = 0
    flag6, flag7 = 0,0
    for v in dict.values():
        sum += v
    rate = 1 -sum
    if x==6:
        flag6 = 1
        dict_rate['X6'] = rate
    else:
        flag7 = 1
        dict_rate['X7'] = rate
    if flag6 == 1:
        dict_rate['X7'] = 0
    if flag7 == 1:
        dict_rate['X6'] = 0
    return dict_rate

if __name__ == "__main__":
    X1_X2(data)
    X3(data)
    X4()
    X5(data)
    X8_X9(data)
    x = x6_x7(data)
    func(x,dict_rate)
    X_rate = pd.Series(dict_rate,index=dict_rate.keys(),name='rate').sort_index()
    print(X_rate)



