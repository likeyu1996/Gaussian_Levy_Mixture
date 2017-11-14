import getdata
import numpy as np
from scipy import stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

log_rate=getdata.get_lograte()
l_mean = np.mean(log_rate)
l_std = np.std(log_rate)
norm=np.random.normal(loc=l_mean,scale=l_std,size=100000)
kstest=stats.kstest(log_rate,'norm')

print('对数收益率均值为'+str(l_mean)+',标准差为'+str(l_std))
print(stats.shapiro(log_rate))
print(stats.normaltest(log_rate))
print(stats.anderson(log_rate, dist='norm'))

sns.distplot(log_rate, bins=100,label='log rate')
sns.distplot(norm,label='norm')
sns.plt.title('mu=%.6f  sigma=%.6f  p=%s'%(l_mean,l_std,kstest[1]))
sns.plt.legend()
sns.plt.show()