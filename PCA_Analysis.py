#%%
import numpy as np
import pandas as pd 
import scipy as sc
import statsmodels.api as stats
import sklearn.decomposition as skd
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.style.use('ggplot')
#%matplotlib inline

#Insert data
yields = pd.read_csv('H:\\JSDev\\Principle_Component_Analysis_Of_Yield_Curve\\data.csv')
#print(yields.head()[0:2])

#Take new subset/trim
yields_subset = yields[['DATA_TYPE_FM', 'TIME_PERIOD','OBS_VALUE']].copy()
yields_subset['TIME_PERIOD'] = pd.to_datetime(yields_subset['TIME_PERIOD'],format='%Y-%m-%d')
yields_subset.set_index('TIME_PERIOD', inplace=True)
yields_subset['DATA_TYPE_FM'].unique()

#print(yields_subset.head())

def extract_type(DATA_TYPE_FM):
    return(DATA_TYPE_FM[0:3])

yields_subset['Yield_Type'] = yields_subset['DATA_TYPE_FM'].apply(extract_type)

yields_spot_rates = yields_subset[yields_subset['Yield_Type'] == 'SR_'].copy()

#Check spot rates
#print(yields_spot_rates.head())

#Apply for all relavant points
yields_3m = yields_spot_rates[yields_spot_rates['DATA_TYPE_FM'] == 'SR_3M'].copy()

#yields_3m['OBS_VALUE'].plot(figsize=(10,6), color = 'green')
#plt.legend((['3 month yield']))
print(yields_3m)
#define yield curve inputs
mats = [1,2,3,5,7,10,20,30] 

time = []; time.append('3M'); time.append('6M')
time_float = []; time_float.append(3/12); time_float.append(6/12)
dataframe_columns = []; dataframe_columns.append('Yield 3M'); dataframe_columns.append('Yield 6M')

#Add year values
for i in mats:
    time.append( str(i) + 'Y')
    time_float.append( i )
    dataframe_columns.append('Yield ' + str(i) + 'Y')

#print(time)
yields = pd.DataFrame( index=yields_3m.index, columns=dataframe_columns)

for i in range(0 , len(time)):
    maturity = time[i]
    yield_name = 'Yield ' + maturity
    yields[yield_name] = yields_spot_rates[yields_spot_rates['DATA_TYPE_FM'] == 'SR_' + maturity]['OBS_VALUE']

print(yields.head())

#Plot yield curve
#figure = plt.gcf(); figure.set_size_inches(16,10.5)
#plt.plot(time_float, yields.iloc[-1], linestyle='-', marker = 'o', color='red', lw = 3)
#plt.plot(time_float, yields.iloc[0], linestyle='--', marker = 'x', color='yellow', lw = 3)
#plt.legend(loc = 'lower right', frameon =True)
#plt.xlabel('Years'); plt.ylabel('Yield')

# plot change in spots
yields.plot( figsize = (16,10))

#Correlation matrix
print(np.round(yields.corr(),2))

#RUN PCA
X = np.matrix(yields)
X_dm = X - np.mean(X, axis = 0)

#Get Eigenvalues
Cov_X = np.cov(X_dm, rowvar = False)
eigen = np.linalg.eig(Cov_X)
eig_values_X = np.matrix(eigen[0])
eig_vectors_X = np.matrix(eigen[1])

#transform data
Y_dm = X_dm * eig_vectors_X

yields_transformed = Y_dm.copy()

#Plot components
plt.figure( figsize = (14,8))
plt.plot( yields_transformed[:, 0:3])
plt.legend(['PC1', 'PC2', 'PC3'])

pc1_yield = yields.copy()
pc1_yield['Yield_PC1'] = yields_transformed[:,0]*(-1)
pc1_yield['Yield_PC2'] = yields_transformed[:,1]
pc1_yield['Yield_PC3'] = yields_transformed[:,2]

pc1_yield[['Yield 1Y','Yield 5Y','Yield 10Y','Yield_PC1']].iloc[:,0:11].plot(figsize=(14,12))

print(np.round(pc1_yield.corr(),2))

#Calculate explained variance
var_explained = np.zeros(eig_values_X.shape[1])
var_explained_agg = np.zeros(eig_values_X.shape[1])

eig_values_X_mat = np.diagflat(np.array(eig_values_X))
eigen_values = eig_values_X_mat.diagonal()
eigen_values_sum_all = np.sum(eigen_values)

for i in range( len(eigen_values)):
    var_explained[i] = eigen_values[i] / eigen_values_sum_all
    eigen_values_sum = np.sum(eigen_values[0:i+1])
    var_explained_agg[i] = eigen_values_sum / eigen_values_sum_all

print('Variance explained: ' , np.round(var_explained[0:5], 2))
print('Aggregate Variance explained: ' , np.round(var_explained_agg[0:5], 2))

#Invert Eigen Vectors

yield_30y = yields.iloc[-1]['Yield 30Y']
print(np.round(yield_30y,2))

yield_30y_pc_123 = yields_transformed[-1,0:3] * eig_vectors_X[-1,0:3].getI()
yield_30y_pc_123 = yield_30y_pc_123 + yields['Yield 30Y'].mean()

print(np.round(yield_30y_pc_123.item(),2))


#Yield Curve Forcasting
n = 10
def calculate_ols_betas(Y, X):  
    nobs = int(X.shape[0])
    nvar = int(X.shape[1])
    betas = (X.getT() * X).getI() * X.getT() * Y
    epsilon = Y - X * betas
    sigma2_epsilon = ((epsilon.getT() * epsilon) / (nobs - nvar)).item()
    var = sigma2_epsilon * (X.getT() * X).getI()
    se = np.sqrt(var.diagonal()).getT()
    tstats = (betas - 0) / se

    return betas, tstats

#Autoregressive estimation
constant = np.empty(3); constant_tstat = np.empty(3)
ar1_coefficient = np.empty(3); ar1_coefficient_tstat = np.empty(3)

for i in range(0,3):
    X = np.matrix(yields_transformed[0:len(yields_transformed)-1,i])
    X = np.insert(X, obj=0, values =1, axis =1)
    Y = np.matrix(yields_transformed[1:len(yields_transformed), i]).reshape(227,1)
    betas, tstats = calculate_ols_betas(Y,X)

    constant[i] = betas[0][0]; constant_tstat[i] = tstats[0][0]
    ar1_coefficient[i] = betas[1][0]; ar1_coefficient_tstat[i] = tstats[1][0]

print('Constant: ', np.round(constant,4)) 
print('t-stats: ', np.round(constant_tstat,4)) 
print('AR(1): ', np.round(ar1_coefficient,4)) 
print('AR(1) t-stats: ', np.round(ar1_coefficient_tstat,4)) 



