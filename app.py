import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import  streamlit as st

data = pd.read_csv(r"G:\pythonproject\emailspam\spam.csv", encoding='ISO-8859-1')
#print(data.head())
#print(data.shape)
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham','span'],['Not spam',['span']])
#print(data.shape)
#print(data.head())
#print(data.isnull().sum())

mess = data['Message']
cato = data['Category']

(mess_train , mess_test,cato_train,cato_test) = train_test_split(mess,cato,test_size=0.2)

cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

#model creation

model =MultinomialNB()
model.fit(features,cato_train)

#test model

features_test = cv.transform(mess_test)
#print(model.score(features_test,cato_test))

#predict
def predict(message):
      input_meassage = cv.transform([message]).toarray()
      result = model.predict(input_meassage)
      return result


st.title('Email Spam Detection')
input_mess = st.text_area('enter your message')

if st.button('Validate'):
    output=predict(input_mess)
    st.write("Prediction:", output[0])
