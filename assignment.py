
"""
Created on Sun May 27 13:42:01 2018

@author: Sushil Surwase
@email: sushilsurwase91@gmail.com
"""
from __future__ import division
import pandas as pd 
import numpy as np

from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


import MySQLdb

# Data base name
dbName = 'servify_assignment'   

 # connecting db to server
dbcon = MySQLdb.connect('52.66.79.237','candidate','asdfgh123',dbName)

#------------------------------------------------------------------------------
# Reading data from SQL server
consumer = "SELECT * FROM consumer;"
consumer_product = "SELECT * FROM consumer_product;"
consumer_servicerequest = "SELECT * FROM consumer_servicerequest;"
plan = "SELECT * FROM plan;"
sold_plan = "SELECT * FROM sold_plan;"

#------------------------------------------------------------------------------
# Storing data to various data frames
consumer = pd.read_sql(consumer, con = dbcon)
consumer_product = pd.read_sql(consumer_product, con = dbcon)
consumer_servicerequest = pd.read_sql(consumer_servicerequest, con = dbcon)
plan = pd.read_sql(plan, con = dbcon)
sold_plan = pd.read_sql(sold_plan, con = dbcon)

#------------------------------------------------------------------------------
# Etracting date, week, month and year from 'DateofPurchase'
sold_plan['Date'] = sold_plan['DateOfPurchase'].dt.date
sold_plan['week'] = sold_plan['DateOfPurchase'].dt.week
sold_plan['month'] = sold_plan['DateOfPurchase'].dt.month
sold_plan['Year'] = sold_plan['DateOfPurchase'].dt.year

#------------------------------------------------------------------------------
ids = sold_plan['week']
Sold_Plans = sold_plan[ids.isin(ids[ids.duplicated()])].sort_values("week")
Total_plans_sold  = len(Sold_Plans.SoldPlanID)
Avg_plans_sold = Total_plans_sold/max(Sold_Plans.week)
print "Average number of plans sold per week: ", Avg_plans_sold

#------------------------------------------------------------------------------

sold_plan = sold_plan.merge(consumer_product[["ConsumerID","BrandID"]],
                            left_on = "ConsumerID", right_on = "ConsumerID", how = "inner")
Brand_A, Brand_B = sold_plan.BrandID.mode()
print "The brand (BrandID) with the highest number of plans bought by customers are: ", Brand_A, Brand_B

#------------------------------------------------------------------------------
Service_requests = consumer_servicerequest.count().values[0]
Plan_sold_count = consumer_servicerequest[consumer_servicerequest.SoldPlanID!=0].count().values[0]
Percentage_service_requests = (Plan_sold_count)/(Service_requests)*100

# Printing msgs
_SERVICE_REQ_RAISED = "The percentage of service requests raised under a plan of the total number of requests raised: "
print _SERVICE_REQ_RAISED, Percentage_service_requests

_PROBABILITY = "The probability that a Service Request will originate against a Sold Plan: "
print _PROBABILITY, (Plan_sold_count)/(Service_requests)

#------------------------------------------------------------------------------
# Data visualization
Consumers_per_week = pd.DataFrame()
Cons_per_week =[]
for week in Sold_Plans['week'].unique():
    Cons_per_week.append(sum(Sold_Plans["week"]==week))
Consumers_per_week["Consumers Per week"] = Cons_per_week

Consumers_per_week.plot(color='red',fontsize=10,legend=None)
plt.grid()
plt.xlabel('Week', {'fontname':'Comic Sans MS'}, fontsize=17)
plt.ylabel('Number of Consumers per Week', {'fontname':'Comic Sans MS'}, fontsize=17)

plt.figure()
Plan_sold_source = sold_plan['Source'].value_counts()
Plan_sold_source.plot(kind='bar')
plt.ylabel('Source-wise Number of Plans Sold', {'fontname':'Comic Sans MS'}, fontsize=17)


# Plotting plan amount and number of consumers
plt.figure()
Plan_sold_source = sold_plan['PlanAmount'].value_counts()
Plan_sold_source.plot(kind='bar')
plt.ylabel('Number of Consumers', {'fontname':'Comic Sans MS'}, fontsize=17)
plt.xlabel('Plan Amount', {'fontname':'Comic Sans MS'}, fontsize=17)

#------------------------------------------------------------------------------
data=pd.DataFrame()
No_of_Con = []
for date in sold_plan["Date"].unique():
    No_of_Con.append(sum(sold_plan["Date"]==date))
data["Number_of_consumer"] = No_of_Con
#data["Plan_amonut"] = sold_plan["PlanAmount"].unique()
data.index=sold_plan["Date"].unique()
data = data.sort_index()

data.plot(color='red',fontsize=10,legend=None)
plt.grid()
plt.xlabel('Date', {'fontname':'Comic Sans MS'}, fontsize=17)
plt.ylabel('Number of Consumers', {'fontname':'Comic Sans MS'}, fontsize=17)


#------------------------------------------------------------------------------
#------------  Time series modeling  ------------------------------------------
predictions = pd.DataFrame()
from statsmodels.tsa.ar_model import AR
model = AR(data)
model_fit = model.fit(10)
predictions['Date_prediction'] = model_fit.predict(start='2017-11-01',end='2018-01-31')
predictions['Num_of_consumers_predicted'] = predictions.Date_prediction.apply(str).str.split('-').str[-1]
predictions["Num_of_consumers_predicted"] = predictions.Num_of_consumers_predicted.apply(float)
predictions.Num_of_consumers_predicted = predictions.Num_of_consumers_predicted.astype(int)

del(predictions['Date_prediction'])
predictions.index = pd.to_datetime(predictions.index)
predictions['month'] = pd.DatetimeIndex(predictions.index).month
Predicted = predictions.groupby([(predictions.index.year),(predictions.index.month)]).sum()
del(Predicted['month'])

#------------------------------------------------------------------------------
data.index = pd.to_datetime(data.index)
in_range_df = data[data.index.isin(pd.date_range("2017-11-01", "2018-01-31"))]
in_range_df['month'] = pd.DatetimeIndex(in_range_df.index).month
in_range_df = in_range_df.groupby([(in_range_df.index.year),(in_range_df.index.month)]).sum()
del(in_range_df['month'])
in_range_df["Num_of_consumers_predicted"] = Predicted["Num_of_consumers_predicted"]

print("Number of Consumers who will join in the months of Nov-17, Dec-17 and Jan-18 are: ")
for t in range(len(Predicted)):
    print('predicted=%f, expected=%f' % ((Predicted.Num_of_consumers_predicted).values[t], 
                                         (in_range_df.Number_of_consumer).values[t]))



#X = data.values.astype(float)
#size = int(len(X) * 0.65)
#train, test = X[0:size], X[size:len(X)] 
#STD_testx=(np.std(test))
#history = [x for x in train]
#predictions = list()
#for t in range(len(test)):
#	model = ARIMA(history,order=(1,0,0))
#	model_fit = model.fit()
#	output = model_fit.forecast(steps=1)
#	yhat = output[0]
#	predictions.append(yhat)
#	obs = test[t]
#	history.append(obs)
#	print('predicted=%f, expected=%f' % (yhat, obs))
#error = (mean_squared_error(test, predictions))**0.5
#print('Test RMSE: %.3f' % error)
#plt.plot(test)
#plt.plot(predictions, color='red')
#plt.show()

