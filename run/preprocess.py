import pandas as pd
import numpy as np
import pickle
import cv2

mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}

df = pd.read_pickle('../data/LSWMD.pkl')
df=df.replace({'failureType':mapping_type})
df.drop(['lotName', 'waferIndex', 'trianTestLabel'],axis=1,inplace=True)

df_withlabel = df[(df['failureType']>=0) & (df['failureType']<=8)]
df_withlabel = df_withlabel[df_withlabel['dieSize'] > 100]

X = df_withlabel['waferMap'].values
y = df_withlabel['failureType'].values.astype(np.int64)

X_binary = np.array([np.where(x==2, 1, 0).astype('uint8') for x in X], dtype=object)
X_resized = np.array([cv2.resize(x*255, (64, 64), interpolation=2) for x in X_binary])

pickle.dump(y, open('../data/y.pickle', 'wb'))
pickle.dump(X_resized, open(f'../data/X_CNN.pickle', 'wb'))