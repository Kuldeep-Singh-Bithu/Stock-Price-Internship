import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
# change  model1  to model
model = pickle.load(open('model1.pkl', 'rb'))
pickle_in=open('news_headline.pickle','rb')
model2=pickle.load(pickle_in)
basicvectorizer=pickle.load(open('vectorizer.pickle','rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(int_features)
    print(final_features)
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    #return render_template('index.html', prediction_text='Stock Price will be going {}'.format(output))
    if(output==-1):
        return render_template('index.html', prediction_text='Stock Price will be going to decrease')
    elif(output==1):
        return render_template('index.html', prediction_text='Stock Price will be going to increase')

####################################
@app.route('/newspred',methods =["GET", "POST"])
def prediction():
    if request.method == "POST":
       news_headline = request.form.get("hdline")
       headline = basicvectorizer.transform([news_headline])
       pred=model2.predict(headline)
       if(pred==1):
           pred="Stock price might increase";
       else:
           pred="Stock price might Decrease"
       return render_template('prediction_page.html',headline=news_headline,Prediction=pred)
    # return render_template('webpage.html',predicted='Harsha')
    return render_template('webpage.html')
# @app.route('/predicted')
if __name__ == "__main__":
    app.run(debug=True)