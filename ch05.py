import os
os.getcwd()

os.chdir('C:\\Users\\Fredrik\\Documents\\pydata-book')

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

obj = Series([4, 7, -5, 3])
obj

obj.values
obj.index

obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2

obj2.index
obj2['a']
obj2['d'] = 6
obj2[['c', 'a', 'd']]
obj2
obj2[obj2 > 0]
obj2 * 2
np.exp(obj2)
'b' in obj2
'e' in obj2

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)
obj3

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)
obj4

pd.isnull(obj4)
pd.notnull(obj4)
obj4.isnull()

obj3
obj4
obj3 + obj4

obj4.name = 'population'
obj4.index.name = 'state'
obj4

obj
obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
obj

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)
frame

DataFrame(data, columns=['year', 'state', 'pop'])

frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                   index=['one', 'two', 'three', 'four', 'five'])
frame2
frame2.columns
frame2['state']
frame2.year

frame2.ix['three']

frame2['debt'] = 16.5
frame2
frame2['debt'] = np.arange(5.)
frame2

val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
frame2

frame2['eastern'] = frame2.state == 'Ohio'
frame2
del frame2['eastern']
frame2.columns

pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = DataFrame(pop)
frame3
frame3.T
DataFrame(pop, index=[2001, 2002, 2003])
pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}
pdata
DataFrame(pdata)
frame3.index.name = 'year'; frame3.columns.name = 'state'
frame3
frame3.values
frame2.values
frame2.index
frame2

obj = Series(range(3), index=['a', 'b', 'c'])
obj
index = obj.index
index
index[1:]
index = pd.Index(np.arange(3))
obj2 = Series([1.5, -2.5, 0], index=index)
obj2.index is index

frame3
'Ohio' in frame3.columns
2003 in frame3.index

obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj

obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj2
obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)

obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3
obj3.reindex(range(6), method='ffill')

frame = DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'],
                  columns=['Ohio', 'Texas', 'California'])
frame
frame2 = frame.reindex(['a', 'b', 'c', 'd'])
frame2
states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)
frame.reindex(index=['a', 'b', 'c', 'd'], method='ffill',
              columns=states)
frame.ix[['a', 'b', 'c', 'd'], states]

obj = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
obj
new_obj = obj.drop('c')
new_obj
obj.drop(['d', 'c'])

data = DataFrame(np.arange(16).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
data
data.drop(['Colorado', 'Ohio'])
data.drop('two', axis=1)
data.drop(['two', 'four'], axis=1)
