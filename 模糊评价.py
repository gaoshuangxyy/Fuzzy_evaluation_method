import numpy as np
import pandas as pd

def s_weights(n,array):
    judge_array = np.array(array).reshape(n, n)
    # print(type(judge_array))
    '''计算权重'''
    #矩阵的特征值和特征向量
    eig1, eig2 = np.linalg.eig(judge_array)
    # print("eig is :")
    # print(eig1)
    # print("eig matrix is :")
    # print(eig2)
    #一致性检验
    '''n=12,ri=1.54'''
    dict_RI = {'1':0,'2':0,'3':0.52,'4':0.89,'5':1.12,'6':1.26,'7':1.36,'8':1.41,
               '9':1.46,'10':1.49,'11':1.52,'12':1.54,'13':1.56,'14':1.58,'15':1.59,}

    ri = dict_RI[str(n)]
    ci = (eig1[0] - n) / (n - 1)
    print('ci is:',ci)
    cr = ci / ri
    print("cr is:",cr)
    if cr < 0.1:
        print('一致性检验通过')
        # sum_num = 0
        mat_weight = np.zeros((n))
        for i in range(n):
            mat_weight[i] = eig2[i][0]
        # print('指标重要程度权重I:\n', mat_weight)
        # 归一化处理
        guiyi_mat_weight = mat_weight / np.sum(mat_weight)
        # print('归一化后指标重要程度权重I：\n', guiyi_mat_weight)
        # print(type(guiyi_mat_weight))
        return mat_weight, guiyi_mat_weight
    else:
        print('一致性检验不通过，调整指标重要性')

def W_weight(array):
    '''结合故障概率，得出指标风险权重集ω={ω1,, ω2, ω3, ω4, ω5, ω6, ω7, ω8, ω9， ω10, ω11, ω12}'''
    #假设S集中的每种故障的概率我们可从故障数据中得出为P
    rate = pd.read_excel('X_rate.xlsx',name = 'x_rate')
    # print(rate)
    rate = list(rate['rate_right'])
    P = np.array([1/3*rate[1],rate[5]+rate[6],rate[7],rate[8],1/2*rate[0],
                1/3*rate[1],1/2*rate[0],1/3*rate[1],rate[2],rate[3],rate[4],rate[5]])
    #W为指标风险权重集ω,并归一化
    W = P*array
    guiyi_W_weight = W / np.sum(W)
    # print('归一化后的指标风险权重集w:\n',guiyi_W)
    return guiyi_W_weight

if __name__ == '__main__':
    '''构建指标集S={S3，S7，S8，S9，S10，S11， S12，S13，S14，S15，S16，S17}的判断矩阵'''
    S_mat_origin = [1, 5, 7, 7, 3, 3, 5, 5, 6, 6, 6, 6,
                  1 / 5, 1, 3, 3, 1 / 5, 1 / 5, 1 / 6, 1 / 6, 1 / 5, 1 / 5, 1 / 5, 1 / 5,
                  1 / 7, 1 / 3, 1, 1, 1 / 5, 1 / 5, 1 / 6, 1 / 6, 1 / 7, 1 / 7, 1 / 7, 1 / 7,
                  1 / 7, 1 / 3, 1, 1, 1 / 5, 1 / 5, 1 / 6, 1 / 6, 1 / 7, 1 / 7, 1 / 7, 1 / 7,
                  1 / 3, 5, 5, 5, 1, 1, 2, 2, 1, 1, 1, 1,
                  1 / 3, 5, 5, 5, 1, 1, 2, 2, 1, 1, 1, 1,
                  1 / 5, 6, 6, 6, 1 / 2, 1 / 2, 1, 1, 1 / 5, 1 / 5, 1 / 5, 1 / 5,
                  1 / 5, 6, 6, 6, 1 / 2, 1 / 2, 1, 1, 3, 3, 3, 3,
                  1 / 6, 5, 7, 7, 1, 1, 5, 1 / 3, 1, 1, 1, 1,
                  1 / 6, 5, 7, 7, 1, 1, 5, 1 / 3, 1, 1, 1, 1,
                  1 / 6, 5, 7, 7, 1, 1, 5, 1 / 3, 1, 1, 1, 1,
                  1 / 6, 5, 7, 7, 1, 1, 5, 1 / 3, 1, 1, 1, 1]
    '''构建每个指标在三种状态下V={正常，异常，故障}的判读矩阵'''
    s3_mat_origin =[1, 5, 7, 1/5, 1, 3, 1/7, 1/3, 1]
    s7_mat_origin =[1, 1/5, 1/7, 5, 1, 1/3, 7, 3, 1]
    s8_mat_origin = [1, 5, 3, 1/5, 1, 1/2, 1/3, 2, 1]
    s9_mat_origin = [1, 2, 3, 1/5, 1, 1/2, 1/3, 2, 1]
    s10_mat_origin = [1, 3, 5, 1/3, 1, 2, 1/5, 1/2, 1]
    s11_mat_origin = [1, 1/4, 1/3, 4, 1, 2, 3, 1/2, 1]
    s12_mat_origin = [1, 1/7, 1/5, 7, 1, 3, 5, 1/3, 1]
    s13_mat_origin = [1, 1/4, 1/6, 4, 1, 1/3, 6, 3, 1]
    s14_mat_origin = [1, 1/5, 1/3, 5, 1, 2, 3, 1/2, 1]
    s15_mat_origin = [1, 1/4, 1/3, 4, 1, 2, 3, 1/2, 1]
    s16_mat_origin = [1, 1/3, 1/2, 3, 1, 2, 2, 1/2, 1]
    s17_mat_origin = [1, 1/3, 1/2, 3, 1, 1, 2, 1, 1]


    s3_mat_weight, s3 = s_weights(3, s3_mat_origin)
    s7_mat_weight, s7 = s_weights(3, s7_mat_origin)
    s8_mat_weight, s8 = s_weights(3, s8_mat_origin)
    s9_mat_weight, s9 = s_weights(3, s9_mat_origin)
    s10_mat_weight, s10 = s_weights(3, s10_mat_origin)
    s11_mat_weight, s11 = s_weights(3, s11_mat_origin)
    s12_mat_weight, s12 = s_weights(3, s12_mat_origin)
    s13_mat_weight, s13 = s_weights(3, s13_mat_origin)
    s14_mat_weight, s14 = s_weights(3, s14_mat_origin)
    s15_mat_weight, s15 = s_weights(3, s15_mat_origin)
    s16_mat_weight, s16 = s_weights(3, s16_mat_origin)
    s17_mat_weight, s17 = s_weights(3, s17_mat_origin)
    s_mat_weight, s_guiyi_mat_weight = s_weights(12, S_mat_origin)
    w_weight = W_weight(s_mat_weight)

    weight = np.concatenate((s3, s7, s8, s9, s10, s11,
                             s12,s13,s14,s15,s16,s17), axis=0).reshape(12,3)

    # print('归一化后的s3的指标得分：\n', s3)
    # print('归一化后的s7的指标得分：\n', s7)
    # print('归一化后的s8的指标得分：\n', s8)
    # print('归一化后的s9的指标得分：\n', s9)
    # print('归一化后的s10的指标得分：\n', s10)
    # print('归一化后的s11的指标得分：\n', s11)
    # print('归一化后的s12的指标得分：\n', s12)
    # print('归一化后的s13的指标得分：\n', s13)
    # print('归一化后的s14的指标得分：\n', s14)
    # print('归一化后的s15的指标得分：\n', s15)
    # print('归一化后的s16的指标得分：\n', s16)
    # print('归一化后的s17的指标得分：\n', s17)
    print('指标得分：\n',weight)
    print('归一化后的指标风险权重集W:\n',w_weight)

    result = np.dot(w_weight,weight)
    print('评判结果：',result)

