import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('parkinsons_disease_detector.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['Jitter(local)', 'Jitter (local_absolute)',
       'Jitter (rap)', 'Jitter (ppq5)', 'Jitter (ddp)', 'Shimmer (local)',
       'Shimmer (local_dB)', 'Shimmer (apq3)', 'Shimmer (apq5)',
       'Shimmer (apq11)', 'Shimmer (dda)', 'AC', 'NTH', 'HTN', 'Median pitch',
       'Mean pitch', 'Standard deviation', 'Minimum pitch', 'Maximum pitch',
       'Number of pulses', 'Number of periods', 'Mean period',
       'Standard deviation of period', 'Fraction of locally unvoiced frames',
       'Number of voice breaks', 'Degree of voice breaks\tstatus', 'UPDRS']
    
    # df = pd.DataFrame(features_value, columns=features_name)
    df = pd.DataFrame(features_value)

    output = model.predict(df)
        
    if output == 0:
        res_val = "** no Parkinson's disease **"
    else:
        res_val = "Parkinson's disease"
        

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
    app.run()
