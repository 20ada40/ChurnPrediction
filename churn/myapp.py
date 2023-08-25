import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/',methods=['GET'])
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    Age = request.form.get('Age')
    Gender = request.form['Gender']
    Location = request.form['Location']
    Subscription_Length_Months = float(request.form['Subscription_Length_Months'])
    Monthly_Bill = float(request.form['Monthly_Bill'])
    Total_Usage_GB = float(request.form['Total_Usage_GB'])

    Average_Monthly_Usage =Total_Usage_GB/Subscription_Length_Months
    Total_Bill =Monthly_Bill*Subscription_Length_Months
    High_Bill_to_Usage_Ratio =(Monthly_Bill/Total_Usage_GB)

    model = pickle.load(open('Model.sav', 'rb'))
    data = [[Age, Gender, Location, Subscription_Length_Months, Monthly_Bill, Total_Usage_GB,Average_Monthly_Usage,Total_Bill,High_Bill_to_Usage_Ratio]]
    df = pd.DataFrame(data, columns=['Age', 'Gender', 'Location', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB','Average_Monthly_Usage','Total_Bill','High_Bill_to_Usage_Ratio'])

    categorical_feature = {feature for feature in df.columns if df[feature].dtypes == 'O'}

    encoder = LabelEncoder()
    for feature in categorical_feature:
        df[feature] = encoder.fit_transform(df[feature])

    single = model.predict(df)
    probability = model.predict_proba(df)[:, 1]
    probability = probability*100

    if single == 1:
        op1 = "This Customer is likely to be Churned!"
        op2 = f"Confidence level is {np.round(probability[0], 2)}"
    else:
        op1 = "This Customer is likely to be Continue!"
        op2 = f"Confidence level is {np.round(probability[0], 2)}"

    return render_template("index.html", op1=op1, op2=op2)


if __name__ == '__main__':
    app.run(debug=True,port=8000)