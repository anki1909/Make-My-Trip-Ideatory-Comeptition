from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn import decomposition,pipeline
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn import ensemble, preprocessing, grid_search, cross_validation
from sklearn import metrics 
from sklearn.calibration import CalibratedClassifierCV

import scipy.stats as scs
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

import scipy as sp


if __name__ == '__main__':
    
    train = pd.read_table('../input/train_search.csv', sep=',')#, nrows = 10000000)
    
    test = pd.read_table('../input/Evaluation.csv', sep=',')
    
    hotels =pd.read_table('../input/Hotel.csv', sep=',')
    
    
    train.columns = ['Search ID', 'Booking Date', 'HotelCode', ' Age', ' Gender',' Number of Rooms', ' Check in date', ' Check Out Date',' Seen Price', ' isClicked', ' isBooked', ' Segment']
    test.columns = ['Search ID', 'Booking Date', 'HotelCode', ' Age', ' Gender',' Number of Rooms', ' Check in date', ' Check Out Date',' Seen Price', ' isClicked', ' isBooked', ' Segment']
    
    
    train = train.merge(hotels, on= 'HotelCode', how = 'left')
    test = test.merge(hotels, on= 'HotelCode', how = 'left')
    
    
    id = test['Search ID'].values
  
    train['Bookingdate_time'] =  pd.to_datetime(train['Booking Date'], format= ' %Y-%m-%d %H:%M:%S')
    test['Bookingdate_time'] =  pd.to_datetime(test['Booking Date'], format= '%d-%m-%Y %H:%M')

    train['year'] = train.Bookingdate_time.dt.year
    train['month'] = train.Bookingdate_time.dt.month
    train['dayofyear'] = train.Bookingdate_time.dt.dayofyear
    train['dayofweek'] = train.Bookingdate_time.dt.dayofweek
    train['day'] = train.Bookingdate_time.dt.day
    
    test['year'] = test.Bookingdate_time.dt.year
    test['month'] = test.Bookingdate_time.dt.month
    test['dayofyear'] = test.Bookingdate_time.dt.dayofyear
    test['dayofweek'] = test.Bookingdate_time.dt.dayofweek
    test['day'] = test.Bookingdate_time.dt.day
    
    
    train['Checkindate_time'] =  pd.to_datetime(train[' Check in date'], format= '%Y-%m-%d')
    test['Checkindate_time'] =  pd.to_datetime(test[' Check in date'], format= '%d-%m-%Y')
    
    
    #train['c_inyear'] = train.Checkindate_time.dt.year
    train['c_inmonth'] = train.Checkindate_time.dt.month
    train['c_indayofyear'] = train.Checkindate_time.dt.dayofyear
    train['c_indayofweek'] = train.Checkindate_time.dt.dayofweek
    train['c_inday'] = train.Checkindate_time.dt.day
    
    #test['c_inyear'] = test.Checkindate_time.dt.year
    test['c_inmonth'] = test.Checkindate_time.dt.month
    test['c_indayofyear'] = test.Checkindate_time.dt.dayofyear
    test['c_indayofweek'] = test.Checkindate_time.dt.dayofweek
    test['c_inday'] = test.Checkindate_time.dt.day
    
    
    
    
    train['Checkoutdate_time'] =  pd.to_datetime(train[' Check Out Date'], format= '%Y-%m-%d')
    test['Checkoutdate_time'] =  pd.to_datetime(test[' Check Out Date'], format= '%d-%m-%Y')
    
    
    #train['c_outyear'] = train.Checkoutdate_time.dt.year
    train['c_outmonth'] = train.Checkoutdate_time.dt.month
    train['c_outdayofyear'] = train.Checkoutdate_time.dt.dayofyear
    train['c_outdayofweek'] = train.Checkoutdate_time.dt.dayofweek
    train['c_outday'] = train.Checkoutdate_time.dt.day
    
    #test['c_outyear'] = test.Checkoutdate_time.dt.year
    test['c_outmonth'] = test.Checkoutdate_time.dt.month
    test['c_outdayofyear'] = test.Checkoutdate_time.dt.dayofyear
    test['c_outdayofweek'] = test.Checkoutdate_time.dt.dayofweek
    test['c_outday'] = test.Checkoutdate_time.dt.day
    
   
    
    
    
    y = train[' Segment'].values
   
    
    text_columns = []
    more_f =  []
    for f in train.columns:
        if train[f].dtype=='object':
            more_f.append(f)
            text_columns.append(f)            
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[f].values) + list(test[f].values))
            train[f] = lbl.transform(list(train[f].values))
            test[f] = lbl.transform(list(test[f].values)) 
            
    
    
    train =train.drop([' Segment','Booking Date',' Check in date', ' Check Out Date','Search ID','Checkoutdate_time','Checkindate_time','Bookingdate_time'], axis =1)
    test =test.drop([' Segment','Booking Date',' Check in date', ' Check Out Date','Search ID','Checkoutdate_time','Checkindate_time','Bookingdate_time'], axis =1)
    
    
    train  = train.replace(np.nan, -1)
    test  = test.replace(np.nan, -1 )
     
    print train.shape,test.shape
    train =np.array(train)
    test = np.array(test)
    train = train.astype(float)
    test = test.astype(float) 
    
   
    
    #0.614773724081
    #tune parameters
    #'max_features': 'sqrt', 'min_samples_split': 5, 'learning_rate': 0.2, 'n_estimators': 100, 'max_depth': 6}

    gbm = ensemble.GradientBoostingClassifier(random_state=42)
    params = [{'n_estimators': [75,100,125], 'min_samples_split': [5,10],'max_depth': [6,8] , 'max_features' : ['sqrt'], 'learning_rate':[0.2]}]    
    clf = grid_search.GridSearchCV(gbm, params, verbose=1,n_jobs = -1)

    # cross validation
    print("k-Fold RMSLE:")
    cv_rmsle = cross_validation.cross_val_score(clf, train, y, scoring='f1')
    print(cv_rmsle)
   
    print("Mean: " + str(cv_rmsle.mean()))

    # get predictions on test
    clf.fit(train, y)

    # get predictions from the model, convert them and dump them!
    preds = clf.predict(test)
    preds = pd.DataFrame({"'Search ID'": id, "cost": preds})
   

    preds.loc[preds.cost == 'backpacker' , 'cost'] = 1
    preds.loc[preds.cost == 'family' , 'cost'] = 2
    preds.loc[preds.cost == 'couple' , 'cost'] = 3
    
    preds['cost'] = preds['cost'].astype(int)
    preds.to_csv('benchmark12.csv', index=False, header =False )

"""    
    [ 0.64105608  0.6339088   0.63522085]
Mean: 0.636728576949
   """
