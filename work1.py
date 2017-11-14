import urllib.request
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
'''from sklearn.mixture import GaussianMixture'''
import levy

def get_page(url): #获取页面数据
    req= urllib.request.Request(url,headers={
        'Connection': 'Keep-Alive',
        'Accept': 'text/html, application/xhtml+xml, */*',
        'Accept-Language':'zh-CN,zh;q=0.8',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko'
        })
    opener= urllib.request.urlopen(req)
    page= opener.read()
    return page

def get_history_data(index,start,end):

    """
    :param index: for example,'sh000001' 上证指数
    :return :
    """
    index_type=index[0:2]
    index_id=index[2:]
    if index_type=='sh':
        index_id='0'+index_id
    if index_type=='sz':
        index_id='1'+index_id
    url ='http://quotes.money.163.com/service/chddata.html?code=%s&start=%s&end=%s&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;VOTURNOVER;VATURNOVER'%(index_id,start,end)

    page=get_page(url).decode('gb2312')
    page=page.split('\r\n')
    col_info=page[0].split(',')
    index_data=page[1:]
    index_data=[x.replace("'",'') for x in index_data]
    index_data=[x.split(',') for x in index_data]


    index_data=index_data[0:index_data.__len__()-1]   #最后一行为空，需要去掉
    pos1=col_info.index('涨跌幅')
    pos2=col_info.index('涨跌额')
    posclose=col_info.index('收盘价')
    index_data[index_data.__len__()-1][pos1]=0     
    index_data[index_data.__len__()-1][pos2]=0
    for i in range(0,index_data.__len__()-1):       
        if index_data[i][pos2]=='None':
            index_data[i][pos2]=float(index_data[i][posclose])-float(index_data[i+1][posclose])
        if index_data[i][pos1]=='None':
            index_data[i][pos1]=(float(index_data[i][posclose])-float(index_data[i+1][posclose]))/float(index_data[i+1][posclose])
    return [index_data,col_info]

data=get_history_data('sh000001','20140331','20170329')
returns=np.log([(float(data[0][i][3]))/float(data[0][i][7]) for i in range((data[0]).__len__()-1,0,-1)])
length=len(returns)
fit_data=st.norm(returns.mean(),returns.std()).rvs(length)
stable_para=levy.fit_levy(returns)
fit_data2=levy.random(alpha=stable_para[0],beta=stable_para[1],mu=stable_para[2],sigma=stable_para[3],shape=(1,length))
'''
returns=returns.reshape((length,1))
aic=[0 for i in range(4)]
bic=[0 for i in range(4)]
for i in range(2,6):
    gmm=GaussianMixture(n_components=i,tol=1e-5,covariance_type='full')
    fgmm=gmm.fit(returns)
    aic[i-2]=fgmm.aic(returns)
    bic[i-2]=fgmm.bic(returns)
k1=int(aic.index(min(aic)))+2
k2=int(bic.index(min(bic)))+2
print (k1)
print (k2)
gmm1=GaussianMixture(n_components=k1,tol=1e-5,covariance_type='full')
gmm2=GaussianMixture(n_components=k2,tol=1e-5,covariance_type='full')
sample1=((gmm1.fit(returns).sample(500))[0].reshape(1,500))
sample2=((gmm2.fit(returns).sample(500))[0].reshape(1,500))
sns.distplot(sample1,bins=500,label='Gaussian1')
sns.distplot(sample2,bins=500,label='Gaussian2')'''
sns.distplot(returns,bins=length,label='Reality')
'''dgmm=gmm.fit(returns).sample(1000)
samples=dgmm[0].reshape((1,1000))
sns.distplot(samples,bins=1000)
sns.distplot(returns,bins=length)
'''
sns.distplot(fit_data,bins=length,label='Norm')
sns.distplot(fit_data2,bins=length,label='Stable')
sns.plt.xlim([-0.15,0.15])
sns.plt.legend()
sns.plt.show()
