import pandas as pd
import numpy as np
import os, glob
from imp_tf import *
from matplotlib import pyplot as plt
from util.params import params
from util.utils import *


# dataset = 'BJTaxi'
# dataset = 'citybike'
# dataset = 'nyctaxi'
preprocess_max = {
    'BJTaxi':1292.0,
    'citybike':526.0,
    'nyctaxi':9806.0
}
# models = ['ResNet', 'UNet', 'ConvLSTM', 'AttConvLSTM', 'ST-Attn', 'ConvLSTM_bi']
models = [
    # 'ResNet',
    # 'UNet',
    # 'UNet_Jump/UNet',
    # 'UNet_Jump/UNet_split',
    # 'ConvLSTM',
    # 'ConvLSTM_bi',
    # 'AttConvLSTM',
    # 'PCRN',
    'ST-Attn',
    'ST-Attn_y',
    'ST-Attn_hm',
    'T-Attn',
    'S-Attn',
    # 'baseline_hm',
    # 'baseline_kde'
]


def loss_mae(xa, xb):
    return np.mean(np.abs(xa - xb))
def sum_mae(xa, xb):
    return np.sum(np.abs(xa - xb))
def loss_mse(xa, xb):
    return np.sqrt(np.mean(np.power(xa - xb, 2)))
def loss_median(xa, xb):
    return np.median(xa - xb)


def jump_last(input, results_path, dataset, split=False):
    a = np.array(input).reshape([-1, params['output_steps']])
    if split == True:
        trgt = np.vstack(np.load(results_path+'UNet_Jump/UNet_split/target.npy'))
        fp = glob.glob(results_path+'UNet_Jump/UNet_split/prediction*.npy')
        fp.sort()
        error = []
        for i in range(int(len(fp)/6)):
            result = []
            for f in fp[i*6:(i+1)*6]:
                result.append(np.vstack(np.load(f)))
            prediction = np.stack(result, axis=1)
            prediction = np.vstack(prediction[-int(trgt.shape[0]/6):])
            error.append(np.sqrt(np.mean(np.square(trgt-prediction))) * preprocess_max[dataset])
            # error.append(np.mean(np.abs(trgt-prediction)) * preprocess_max[dataset])
        # return list(np.mean(a, axis=1))
        return error
    else:
        return list(a[:,-1])

def whole_performance(results_path, dataset):
    results_dict = {}
    for m in models:
        results_dict[m] = {
            'train_loss':[],
            'val_loss':[],
            'test_loss':[]
        }
        fp = glob.glob(results_path+m+'/*.csv')
        fp.sort()
        for f in fp:
            data = pd.read_csv(f)
            results_dict[m]['train_loss'].append(data[data['loss_type']=='train_loss']['loss'].values[-1])
            results_dict[m]['val_loss'].append(data[data['loss_type']=='val_loss']['loss'].values[-1])
            results_dict[m]['test_loss'].append(data[data['loss_type']=='test_loss']['loss'].values[-1])
        if 'Jump' in m:
            results_dict[m]['train_loss'] = jump_last(results_dict[m]['train_loss'], results_path, dataset, split='split' in m)
            results_dict[m]['val_loss'] = jump_last(results_dict[m]['val_loss'], results_path, dataset, split='split' in m)
            results_dict[m]['test_loss'] = jump_last(results_dict[m]['test_loss'], results_path, dataset, split='split' in m)


    train_loss = [np.array(results_dict[k]['train_loss'], dtype=float) for k in results_dict]
    val_loss = [np.array(results_dict[k]['val_loss'], dtype=float) for k in results_dict]
    test_loss = [np.array(results_dict[k]['test_loss'], dtype=float) for k in results_dict]
    keys = [item for item in results_dict]
    # plt.figure(figsize=(15,6))
    # plt.boxplot(train_loss, labels=keys)
    # plt.grid()
    # plt.show()
    # plt.figure(figsize=(15,6))
    # plt.boxplot(val_loss, labels=keys)
    # plt.grid()
    # plt.show()
    plt.figure(figsize=(15,6))
    plt.boxplot(test_loss, labels=keys)
    plt.grid()
    plt.show()
    return [np.round(np.array(results_dict[k]['test_loss'], dtype=float).mean(),3) for k in results_dict]


def each_step(results_path, dataset):
    results = []
    for m in models:
        # m = 'ConvLSTM'
        print(m)
        trgt = np.vstack(np.load(results_path + 'ST-Attn/target.npy'))
        fp = glob.glob(results_path + m + '/prediction*.npy')
        fp.sort()
        if 'Jump' in m and 'split' not in m:
            fp = np.array(fp).reshape([-1, 6])[:, -1].tolist()
        pred = []

        for f in fp:
            if m in ['ResNet', 'UNet']:
                pred.append(np.load(f))
            elif m == 'PCRN':
                pred.append(np.transpose(np.load(f), [1,2,3,4,0]))
            elif 'Jump' in m and 'split' not in m:
                pred.append(np.vstack(np.load(f)))
            elif 'Jump' in m and 'split' in m:
                pred.append(np.vstack(np.load(f)))
            else:
                pred.append(np.vstack(np.load(f)))
        if 'Jump' in m and 'split' in m:
            predtmp = []
            for i in range(int(len(pred)/6)):
                predtmp.append(np.stack(pred[i*6:(i+1)*6], axis=1))
            pred = predtmp
        # pred = np.mean(pred, axis=0)
        PRECISION = []
        for item in pred:
            xshape = item.shape
            tslots = np.min([xshape[0], trgt.shape[0]])
            precision = []
            for i in range(6):
                if m == 'PCRN':
                    precision.append(np.sqrt(np.mean(np.square(trgt[-tslots:, i, :,:,:]* preprocess_max[dataset]-item[-tslots:, i, :,:,:]))))
                else:
                    precision.append(
                        np.sqrt(np.mean(np.square(trgt[-tslots:, i, :,:,:] - item[-tslots:, i, :,:,:]))) * preprocess_max[dataset])
            PRECISION.append(precision)
        precision = np.round(np.mean(PRECISION, axis=0),3)
        results.append(precision)
    results_df = pd.DataFrame(results, columns=['step'+str(i+1) for i in range(6)])
    results_df.to_csv('../result-collect/'+dataset+'_self.csv', index=False)


def abnormal_test(results_path, dataset):
    if dataset in ['BJTaxi', 'citybike', 'nyctaxi']:
        hm_ext = kde_data2(dataset)
    else:
        hm_ext = hm_nycdata(dataset='citybike')
    _, test_hm = batch_data(data=hm_ext, batch_size=FLAGS.batch_size,
                            input_steps=FLAGS.input_steps, output_steps=FLAGS.output_steps)
    test_hm = np.vstack(test_hm[-20:])

    results = []
    for m in models:
        # m = 'baseline_hm'
        print(m)
        trgt = np.vstack(np.load(results_path + 'ST-Attn/target.npy'))
        fp = glob.glob(results_path + m + '/predict*.npy')
        fp.sort()
        if 'Jump' in  m and 'split' not in m:
            fp = np.array(fp).reshape([-1, 6])[:, -1].tolist()
        pred = []

        for f in fp:
            if m in ['ResNet', 'UNet']:
                pred.append(np.load(f))
            elif m == 'PCRN':
                pred.append(np.transpose(np.load(f), [1,2,3,4,0]))
            elif 'Jump' in m and 'split' not in m:
                pred.append(np.vstack(np.load(f)))
            elif 'Jump' in m and 'split' in m:
                pred.append(np.vstack(np.load(f)))
            elif 'baseline' in m:
                pred.append(np.load(f))
            else:
                pred.append(np.vstack(np.load(f)))
        if 'Jump' in m and 'split' in m:
            predtmp = []
            for i in range(int(len(pred)/6)):
                predtmp.append(np.stack(pred[i*6:(i+1)*6], axis=1))
            pred = predtmp
        # pred = np.mean(pred, axis=0)
        PRECISION = []
        for item in pred:

            xshape = item.shape
            tslots = np.min([xshape[0], trgt.shape[0]])
            abnormals = np.abs(trgt[-tslots:]-test_hm[-tslots:]) / (test_hm[-tslots:]+1)
            abnormals = np.where(abnormals < 0.5, 0, 1)
            precision = []
            for i in range(6):
                if m == 'PCRN':
                    precision.append(np.sqrt(1/abnormals[:,i].sum()*np.sum(abnormals[:,i]*np.square(trgt[-tslots:, i, :,:,:]* preprocess_max[dataset]-item[-tslots:, i, :,:,:]))))
                elif 'baseline' in m:  
                    precision.append(
                        np.sqrt(1/abnormals[:,i].sum()*np.sum(abnormals[:,i]*np.square(trgt[-tslots:, i, :,:,:]* preprocess_max[dataset] - item[-tslots:, i, :,:,:]))) )
                else:
                    precision.append(
                        np.sqrt(1/abnormals[:,i].sum()*np.sum(abnormals[:,i]*np.square(trgt[-tslots:, i, :,:,:]* preprocess_max[dataset] - item[-tslots:, i, :,:,:]))) )
            PRECISION.append(precision)
        precision = np.round(np.mean(PRECISION, axis=0),3)
        results.append(precision)
    results_df = pd.DataFrame(results, columns=['step'+str(i+1) for i in range(6)])
    results_df.to_csv('../result-collect/'+dataset+'_abnormal.csv', index=False)


def instance(results_path, dataset):
    results = []
    for m in models:
        # m = 'ConvLSTM'
        print(m)
        trgt = np.vstack(np.load(results_path + 'ST-Attn/target.npy'))
        fp = glob.glob(results_path + m + '/prediction*.npy')
        fp.sort()
        if 'Jump' in m and 'split' not in m:
            fp = np.array(fp).reshape([-1, 6])[:, -1].tolist()
        pred = []

        for f in fp:
            if m in ['ResNet', 'UNet']:
                pred.append(np.load(f))
            elif m == 'PCRN':
                pred.append(np.transpose(np.load(f), [1,2,3,4,0]))
            elif 'Jump' in m and 'split' not in m:
                pred.append(np.vstack(np.load(f)))
            elif 'Jump' in m and 'split' in m:
                pred.append(np.vstack(np.load(f)))
            else:
                pred.append(np.vstack(np.load(f)))
        if 'Jump' in m and 'split' in m:
            predtmp = []
            for i in range(int(len(pred)/6)):
                predtmp.append(np.stack(pred[i*6:(i+1)*6], axis=1))
            pred = predtmp
        # pred = np.mean(pred, axis=0)
        trgt_12 = np.concatenate([trgt[-week_len+24:-week_len+24+7,:,16,16,0][:-1,0],
                                 trgt[-week_len+24:-week_len+24 + 7, :, 16, 16, 0][-1, :]]) * preprocess_max[dataset]
        pred_12 = np.concatenate([trgt[-week_len+24:-week_len+24+7,:,16,16,0][:-1,0],
                                 pred[0][-week_len+24:-week_len+24 + 7, :, 16, 16, 0][-1, :]])
        if m != 'PCRN':
            pred_12 = pred_12 * preprocess_max[dataset]
        results.append(pred_12)
    results.append(trgt_12)
    a = np.array(results)
    plt.plot(a.transpose())
    plt.show()


def error_distribute():
    pre_path = '../BJTaxi-results/results/ST-Attn_y_poe/'
    target = np.load(pre_path + 'target.npy')
    pre_file = [
        'prediction08150603.npy',
        'prediction08151028.npy',
        'prediction08151324.npy',
        'prediction08170142.npy',
        'prediction06160153.npy'
    ]
    for item in pre_file:
        if item[10:12] == '08':# curve trans
            predict = np.load(pre_path + item)
        else: # forward
            predict = np.load(pre_path + item) * 1292
            predict = np.concatenate(predict)
        ERROR = []
        interval = 5
        for i in range(1292//interval+1):
            # _m = np.where(target==i, 1, 0)
            _m = np.where(target<(i+1)*interval, 1, 0)
            # _m = np.where(target>i*interval, 1, 0)
            _t = np.multiply(target, _m)
            _p = np.multiply(predict, _m)
            # error = np.sum(np.abs(_t - _p)) / _m.sum()
            error = np.sum(np.abs(_t - _p))
            ERROR.append(error)
        plt.plot(ERROR, label=item)
    plt.legend()
    plt.show()
    print('end')


def error_dis():
    dataset = 'BJTaxi'
    prepath = '/cluster/zhouyirong09/peer-work/ST-Attn/'+dataset+'-results/results/ST-Attn_y_poe/'
    a = (np.concatenate(np.load(prepath+'prediction'+'06160153.npy'))).flatten()
    b = (np.load(prepath+'validation08230443.npy')).flatten()
    c = (np.load(prepath+'prediction'+'08240055.npy')).flatten()
    d = (np.load(prepath+'validation08240055.npy')).flatten()
    e1 = (np.load(prepath+'valt.npy')).flatten()
    e2 = (np.load(prepath+'target.npy')).flatten()

    print('minmax loss-predict:', loss_mae(a, e2))
    print('minmax loss-validate:', loss_mae(b, e1))
    print('hist_0 loss-predict:', loss_mae(c, e2))
    print('hist_0 loss-validate:', loss_mae(d, e1))
    idx = np.argwhere(e2 == 0)
    idx1 = np.argwhere(e2 < 10)
    idelta = np.array(list(set(idx1.flatten()) - set(idx.flatten())))
    idelta = idelta.reshape([idelta.shape[0],1])
    idx2 = np.argwhere(e2 >= 10)
    x = np.concatenate([np.abs(a[idx] - e2[idx]), np.abs(a[idx2] - e2[idx2]), np.abs(c[idelta] - e2[idelta])])
    y = np.concatenate([np.abs(a[idx] - e2[idx]), np.abs(a[idx2] - e2[idx2]), np.abs(a[idelta] - e2[idelta])])
    print('merge loss-validate:', np.mean(x))

    # c = (np.concatenate(np.load(prepath+'prediction06160642.npy')) * 1294).flatten()
    # b = (np.load(prepath+'valt.npy')).flatten()
    e1 = e1.astype('int')
    e2 = e2.astype('int')

    eidx1 = np.argsort(e1)
    eidx2 = np.argsort(e2)
    a = a[eidx1]
    b = b[eidx2]
    c = c[eidx1]
    d = d[eidx2]
    e1 = e1[eidx1]
    e2 = e2[eidx2]
    ax = []
    bx = []
    cx = []
    dx = []
    for i in range(0,1000):
    # for i in range(e1[-1]):
        eidx1 = np.argwhere(e1 == i)
        bx.append(sum_mae(b[eidx1], i))
        dx.append(sum_mae(d[eidx1], i))
    # for i in range(e2[-1]):
    for i in range(0,1000):
        eidx2 = np.argwhere(e2 == i)
        ax.append(sum_mae(a[eidx2], i))
        cx.append(sum_mae(c[eidx2], i))
    plt.plot(ax, label='a')
    plt.plot(cx, label='c')
    plt.plot(bx, label='b')
    plt.plot(dx, label='d')
    plt.legend()
    plt.grid()
    plt.show()



def stdv_analyst():
    dataset = 'BJTaxi'
    if dataset == 'BJTaxi':
        data, train_data, val_data, test_data, all_timestamps = \
            load_BJdata(fpath='../../PCRN/data/TaxiBJ-filt/', split=split_point, T=params['ts_oneday'])
        hm_ext = hm_BJTaxi(pre_path='../')
        kde_ext = kde_data2(dataset=dataset, pre_path='../')
    else:
        data, train_data, val_data, test_data = \
            load_npy_data(filename=['../data/'+dataset+'/p_map.npy', '../data/'+dataset+'/d_map.npy'], split=split_point)
        hm_ext = hm_nycdata(dataset=dataset, pre_path='../')
        kde_ext = kde_data2(dataset=dataset, pre_path='../')

    kx = []
    hx = []
    data = data.flatten()
    hm_ext = hm_ext.flatten()
    kde_ext = kde_ext.flatten()

    for i in range(0,1000):
    # for i in range(e1[-1]):
        eidx1 = np.argwhere(data == i)
        kx.append(loss_mse(kde_ext[eidx1], i))
        hx.append(loss_mse(hm_ext[eidx1], i))
        # kx.append(kde_ext[eidx1]-i)
        # hx.append(hm_ext[eidx1]-i)
    plt.plot(kx, label='k')
    plt.plot(hx, label='h')
    plt.legend()
    plt.grid()
    plt.show()


def linearI_transform(data, xmin, xmax):
    data = 1. * data * (xmax - xmin) + xmin
    return data
def linearT_transform(data, xmin, xmax):
    data = 1. * (data - xmin) / (xmax - xmin)
    return data
def use_bjdata():
    data, train_data, val_data, test_data, all_timestamps = \
        load_BJdata(fpath='../../PCRN/data/TaxiBJ-filt/', split=split_point, T=params['ts_oneday'])

    data = test_data.flatten().astype('int')

    k = 1.2
    m = 0
    e = data + m
    xrnd = np.random.standard_normal(e.shape[0]) * k
    # x = np.add(xrnd, e)# + e
    x = np.multiply(xrnd, e+1) + e
    # x = k * (data+1) + data

    pre_process = Histogram_Normalize('hist_2')
    pre_process.fit(e)
    y = pre_process.transform(e)
    y = pre_process.cdf_line(y)
    y = -np.abs(np.multiply(xrnd, y+1)) + y
    # y = -np.abs(xrnd) + y
    y = pre_process.cdf_rline(y)
    y = np.abs(y)
    y = pre_process.inverse_transform(y)

    ax = []
    bx = []
    for i in range(0, e.max()):
        eidx = np.argwhere(e == i)
        ax.append(sum_mae(x[eidx], i))
        bx.append(sum_mae(y[eidx], i))
        # plt.hist(x[eidx], bins=100, alpha=0.4)
        # plt.hist(y[eidx], bins=100, alpha=0.4)
        # plt.show()
    # plt.xlim(1,20)
    # plt.ylim(0, 20000)
    plt.plot(ax, label='a')
    plt.plot(bx, label='b')
    plt.legend()
    plt.grid()
    plt.show()
    return e, x



def use_bjdata1():
    k = 0.1
    m = 0
    e = data + m
    xrnd = np.random.standard_normal(e.shape[0]) * k
    x = np.multiply(xrnd, e+1) + e
    # x = k * (data+1) + data

    pre_process = CurveTrans()
    pre_process.fit(e)
    y = pre_process.transform(e)
    y = linearI_transform(y, m, 100+m)
    yrnd = np.random.standard_normal(e.shape[0]) * k
    y = np.multiply(yrnd, y+1) + y
    # y = k * (y+1) + y
    y = linearT_transform(y, m, 100+m)
    y = pre_process.inverse_transform(y)

    ax = []
    bx = []
    for i in range(e.max()):
        eidx = np.argwhere(e == i)
        ax.append(sum_mae(x[eidx], i))
        bx.append(sum_mae(y[eidx], i))
        # plt.hist(x[eidx], bins=100, alpha=0.4)
        # plt.hist(y[eidx], bins=100, alpha=0.4)
        # plt.show()
    plt.plot(ax[m:], label='a')
    plt.plot(bx[m:], label='b')
    plt.legend()
    plt.grid()
    plt.show()
    return e, x


def use_bjdata2():
    _, _, _, data = load_station_data(filename='../data/citybike/data.npy',
                                      split=split_point)
    data = data.flatten().astype('int')
    # data, _ = generate_x()
    # data = data.astype('int')

    k = 0.05
    m = 0
    e = data + m
    xrnd = np.random.standard_normal(e.shape[0]) * k
    x = np.add(xrnd, e)# + e
    # x = np.multiply(xrnd, e+0.1) + e
    # x = k * (data+1) + data

    pre_process = Histogram_Normalize('hist_2')
    pre_process.fit(e)
    y = pre_process.transform(e)
    e_p = pre_process.cdf_line(pre_process.cdf)
    e_p = e_p - k
    e_p[0] += 2*k
    e_p = pre_process.cdf_rline(e_p+k)
    e_p = pre_process.inverse_transform(e_p)
    e_p = e_p - range(109)
    c_n = [np.argwhere(e == i).shape[0] for i in range(e.max()+1)]
    bx = np.multiply(np.abs(e_p), c_n)

    ax = []
    # bx = []
    for i in range(0, e.max()):
        eidx = np.argwhere(e == i)
        ax.append(sum_mae(x[eidx], i))
        # bx.append(sum_mae(y[eidx], i))
        # plt.hist(x[eidx], bins=100, alpha=0.4)
        # plt.hist(y[eidx], bins=100, alpha=0.4)
        # plt.show()
    # plt.xlim(1,20)
    # plt.ylim(0, 20000)
    plt.plot(ax, label='a')
    plt.plot(bx, label='b')
    plt.legend()
    plt.grid()
    plt.show()
    return e, x


def use_bjdata3():
    _, _, _, data = load_station_data(filename='../data/citybike/data.npy',
                                      split=split_point)
    data = data.flatten().astype('int')
    # data, _ = generate_x()
    # data = data.astype('int')

    k = 0.1
    m = 0
    e = data + m
    xrnd = np.random.standard_normal(e.shape[0]) * k
    # x = np.add(xrnd, e)# + e
    x = np.multiply(xrnd, e+1) + e
    # x = k * (data+1) + data

    pre_process = Histogram_Normalize('hist_2')
    pre_process.fit(e)
    y = pre_process.transform(e)
    y = pre_process.cdf_line(y)
    y = np.multiply(xrnd, y+1) + y
    y = pre_process.cdf_rline(y)
    y = pre_process.inverse_transform(y)

    ax = []
    bx = []
    for i in range(0, e.max()):
        eidx = np.argwhere(e == i)
        ax.append(sum_mae(x[eidx], i))
        bx.append(sum_mae(y[eidx], i))
        # plt.hist(x[eidx], bins=100, alpha=0.4)
        # plt.hist(y[eidx], bins=100, alpha=0.4)
        # plt.show()
    # plt.xlim(1,20)
    # plt.ylim(0, 20000)
    plt.plot(ax, label='a')
    plt.plot(bx, label='b')
    plt.legend()
    plt.grid()
    plt.show()
    return e, x


if __name__ == '__main__':
    # use_bjdata()
    # stdv_analyst()
    # error_dis()
    results = []
    for dataset in ['BJTaxi', 'nyctaxi', 'citybike']:
        results_path = '/cluster/zhouyirong09/peer-work/ST-Attn/result-collect/' + dataset + '/'
        results.append(whole_performance(results_path, dataset))

    result_df = pd.DataFrame(results, columns=models)
    result_df.to_csv('../result-collect/average_self.csv', index=False)

    results = []
    for dataset in ['BJTaxi', 'nyctaxi', 'citybike']:#
        results_path = '/cluster/zhouyirong09/peer-work/ST-Attn/result-collect/' + dataset + '/'
        # instance(results_path, dataset)
        # abnormal_test(results_path, dataset)
        each_step(results_path, dataset)