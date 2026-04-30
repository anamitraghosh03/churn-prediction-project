

# %%
import requests

# %%
host = 'churn-serving-env.eba-cfuqxid3.eu-north-1.elasticbeanstalk.com'
url = f'http://{host}/predict'

# %%
customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": (1*29.85)
}

# %%
customer

# %%
response = requests.post(url, json = customer).json()
print(response)

# %%
if response['churn']==True:
    print('sending promo email to %s' %('xyz-123'))

# %%



