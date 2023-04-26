# import libraries
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# reading in the dataset
df = pd.read_csv('Iris.csv')
df.head()

# describing the data
df.describe().T

# splitting to train and test set
X = df.drop(['Id', 'Species'], axis =1)
y = df['Species']

# splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# initialize the model
m1 = RandomForestClassifier()

# train the model
m1.fit(X_train, y_train)

# test the model
ypred1 = m1.predict(X_test)

# accuracy score
acc1 = accuracy_score(y_test, ypred1)
print(acc1)

# create function for gradio app to get the predictions
def get_predictions(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm):
    x = np.array([SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm])
    preds = m1.predict(x.reshape(1,-1))
    return preds

# creating the gradio interface
outputs = gr.outputs.Textbox()
demo = gr.Interface(fn=get_predictions, inputs=['number', 'number', 'number', 'number'], outputs=outputs,
                   description = 'This a classification Iris model using Random forest')
demo.launch()
