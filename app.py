
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import streamlit as st

# Load dataset
data = pd.read_csv(r"G:\pythonproject\emailspam\spam.csv", encoding='ISO-8859-1')

# Preprocess
data.drop_duplicates(inplace=True)
data = data.rename(columns={data.columns[0]: 'Category', data.columns[1]: 'Message'})  # Rename first two columns
data['Category'] = data['Category'].replace({'ham': 'Not Spam', 'spam': 'Spam'})
data = data[['Category', 'Message']]  # Only keep necessary columns

# Prepare features
mess = data['Message']
cato = data['Category']

mess_train, mess_test, cato_train, cato_test = train_test_split(mess, cato, test_size=0.2, random_state=42)

cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

# Train model
model = MultinomialNB()
model.fit(features, cato_train)

# Evaluate model
features_test = cv.transform(mess_test)
accuracy = model.score(features_test, cato_test)
predictions_test = model.predict(features_test)
cm = confusion_matrix(cato_test, predictions_test)

# Prediction function
def predict(message):
    input_message = cv.transform([message])
    result = model.predict(input_message)
    return result[0]

# Bulk prediction function
def predict_bulk(messages):
    features_bulk = cv.transform(messages)
    return model.predict(features_bulk)

# Streamlit app layout
st.title('Email Spam Detection App')

# Single message prediction
st.subheader(" Check a Single Message")
input_mess = st.text_area('Enter your message here:')

if st.button('Validate'):
    if input_mess.strip() != "":
        result = predict(input_mess)
        if result == "Spam":
            st.error("This message is likely Spam.")
        else:
            st.success("This message is Not Spam.")
    else:
        st.warning("Please enter a message before validating.")

# Display accuracy
st.subheader("Model Performance")
st.write(f"Accuracy: {accuracy * 100:.2f}%")

# Show confusion matrix
st.write("### Confusion Matrix")
fig, ax = plt.subplots()
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_).plot(ax=ax)
st.pyplot(fig)
# Bulk prediction via CSV

st.subheader("Bulk Prediction - Upload CSV")
uploaded_file = st.file_uploader("Upload a CSV file containing messages (optionally with labels)", type=["csv"])

if uploaded_file:
    try:
        # Read and clean
        new_data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        new_data = new_data.loc[:, ~new_data.columns.str.contains('^unnamed', case=False)]
        new_data.columns = new_data.columns.str.strip().str.lower()

        # Detect message column
        possible_msg_cols = ['message', 'text', 'content', 'body', 'email', 'v2']
        message_col = next((col for col in possible_msg_cols if col in new_data.columns), None)

        if not message_col:
            message_col = st.selectbox("Select the column containing messages:", new_data.columns)

        st.info(f"Using message column: {message_col}")

        # Detect actual label column
        label_col = None
        possible_label_cols = ['category', 'label', 'target', 'v1']
        for col in possible_label_cols:
            if col in new_data.columns:
                label_col = col
                break

        if st.button("Run Predictions on Uploaded Data"):
            predictions = predict_bulk(new_data[message_col])
            new_data['Prediction'] = predictions

            st.success(" Predictions completed!")
            st.write(new_data)

            # Show confusion matrix if actual labels are available
            if label_col:
                st.info(f"Detected actual label column:{label_col}")

                # Normalize the actual labels
                actual = new_data[label_col].str.strip().str.lower().replace({'spam': 'Spam', 'ham': 'Not Spam'})
                predicted = new_data['Prediction']

                cm = confusion_matrix(actual, predicted, labels=['Spam', 'Not Spam'])

                st.write(" Confusion Matrix (Actual vs Predicted)")
                fig, ax = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Spam', 'Not Spam'])
                disp.plot(ax=ax)
                st.pyplot(fig)
            else:
                st.warning("No label column found to show a confusion matrix.")

            # Downloadable results
            csv = new_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")
