import tushare as ts
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

import urllib.request
import time

#通过tushare获取上证指数数据(数据太少而且无法自由选定时间，已弃用)
'''
h_sh=ts.get_hist_data('sh')
#h_sh.to_csv('hist of sh.csv')
#h_sh = pd.read_csv('hist of sh.csv')
price = h_sh['close']
l_p=len(price)

#获取对数收益率
def get_lograte():
    lrate=[0 for i in range(l_p)]
    for i in range(l_p):
        if i < l_p - 1:
            lrate[i] = math.log(price[i]) - math.log(price[i + 1])
        else:
            lrate[i] = None
    log_rate=pd.DataFrame(lrate[:len(lrate) - 1],columns=['log rate'])
    return log_rate

#获取简单收益率
def get_simplerate():
    rate=[0 for i in range(l_p)]
    for i in range(l_p):
        if i < l_p - 1:
            rate[i] = (price[i]- price[i + 1])/price[i + 1]
        else:
            rate[i] = None
    simple_rate=pd.DataFrame(rate[:len(rate) - 1],columns=['simple rate'])
    return simple_rate
'''

#爬虫获取数据
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

def get_lograte(index='sh000001',start='20140331',end='20170329'):
    data=get_history_data(index,start,end)
    returns=np.log([float(data[0][i][3])/float(data[0][i][7]) for i in range((data[0]).__len__()-1,0,-1)])
    return returns

def get_simplerate(index='sh000001',start='19910101',end='20170329'):
    data=get_history_data(index,start,end)
    returns=[(float(data[0][i][3])/float(data[0][i][7])-1) for i in range((data[0]).__len__()-1,0,-1)]
    return returns

if __name__ == "__main__":
    l_r=pd.DataFrame(get_lograte(),columns=['log rate'])
    l_r.to_csv('log rate of sh14.csv')
'''    data=get_history_data('sh000001','19961216','20170329')
    col_name=data[1]
    data_961216=pd.DataFrame(data[0],columns=col_name)
    data_961216.to_csv('all hist of sh.csv')'''