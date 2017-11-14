import getdata
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

log_rate=getdata.get_lograte()
length=len(log_rate)
log_rate_array=np.array(log_rate)
log_rate_array_n=log_rate_array.reshape(length,1)

#赤池信息准则
aic_n=[0 for i in range(5)]
#贝叶斯信息准则
bic_n=[0 for i in range(5)]

for i in range(2,7):
    gmm_rate=GaussianMixture(n_components=i,tol=1e-5,covariance_type='full')
    gmm_rate_fit=gmm_rate.fit(log_rate_array_n)
    aic_n[i-2]=gmm_rate_fit.aic(log_rate_array_n)
    bic_n[i-2]=gmm_rate_fit.bic(log_rate_array_n)   

aic_min=aic_n.index(min(aic_n))+2
bic_min=bic_n.index(min(bic_n))+2

def sample_a(n=100000):
    gmm_a=GaussianMixture(n_components=aic_min,tol=1e-5,covariance_type='full')
    sample_a=(gmm_a.fit(log_rate_array_n).sample(n))[0].reshape(n)
    return sample_a

def sample_b(n=100000):
    gmm_b=GaussianMixture(n_components=bic_min,tol=1e-5,covariance_type='full')
    sample_b=(gmm_b.fit(log_rate_array_n).sample(n))[0].reshape(n)
    return sample_b

a=sample_a()
b=sample_b()
ks_a=stats.ks_2samp(log_rate,a)
ks_b=stats.ks_2samp(log_rate,b)
sns.distplot(log_rate,bins=100,label='log rate')
sns.distplot(a,bins=100,label='Gaussianmixture_aic')
sns.distplot(b,bins=100,label='Gaussianmixture_bic')

sns.plt.title('aic=%d  bic=%d     p_a=%.4f p_b=%.4f'%(aic_min,bic_min,ks_a[1],ks_b[1]))
sns.plt.legend()
sns.plt.show()








