import pandas as pa

raw_data = pa.read_csv("../Data/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

#print(raw_data)
# Process fields
'''
Yes = 1, No = 0
Male = 1, Female = 0
No phone service = 2 (Since No is 0 and Yes is 1, 2 is left)
DSL = 1, Fiber optic = 2, (No = 0)
No internet service = 2 (Since No is 0 and Yes is 1, 2 is left)

--Contracts--
Month-to-month = 0
One year = 1
Two year = 2

-- Payment Methods
Mailed check = 0
Electronic check = 1
Credit card = 2
Bank transfer = 3
'''
raw_data.replace(('Yes', 'No', 'Male', 'Female', 'No phone service', 'DSL', 'Fiber optic', 'No internet service', 'Month-to-month', 'One year', 'Two year', 'Mailed check', 'Electronic check', 'Credit card (automatic)', 'Bank transfer (automatic)')
                 , (1,    0,     1,       0,               2,          1,       2,              2,                        0,            1,            2,          0,             1,                   2,             3), inplace=True)
print(raw_data)
pa.DataFrame.to_csv(raw_data, "../Data/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn_Processed.csv")