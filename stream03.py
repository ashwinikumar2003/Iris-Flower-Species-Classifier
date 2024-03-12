import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

iris = pd.read_csv(r"C:\Users\Ashwini\Downloads\Iris.csv")

st.write('''
# Iris Flower Species Classifier
this app classifies iris flower species based upon certain parameters
''')
st.sidebar.header("User input parameters")

def user_inputs():
    sepal_length = st.sidebar.slider("Sepal Length (in cm)", 4.0, 8.0, 6.0)
    sepal_width = st.sidebar.slider("Sepal Width (in cm)", 2.0, 5.0, 3.0)
    petal_length = st.sidebar.slider("Petal Length (in cm)", 1.0, 7.0, 3.0)
    petal_width = st.sidebar.slider("Petal Width (in cm)", 0.1, 3.0, 1.0)

    data = {'SepalLengthCm':sepal_length, 'SepalWidthCm':sepal_width, 'PetalLengthCm':petal_length, 'PetalWidthCm':petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_inputs()
st.subheader("User Input Parameters")
st.write(df)

class_images = {
    0: [r"C:\Users\Ashwini\Downloads\Iris_setosa.jpg", "Iris Setosa"],
    1: [r"C:\Users\Ashwini\Downloads\Iris_versicolor_3.jpg", "Iris Versicolor"],
    2: [r"C:\Users\Ashwini\Downloads\Iris_virginica.jpg", "Iris Verginica"]
}


classify = RandomForestClassifier(100)
x=iris.drop(['Id', 'Species'], axis=1)
y=(LabelEncoder().fit_transform(iris['Species'])).reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7)
classify.fit(x_train, y_train)

st.subheader("Flower Species")
y_pred = classify.predict(df)
if y_pred[0] in class_images:
    st.image(class_images[y_pred[0]][0], caption=class_images[y_pred[0]][1], width=400)


st.write(classify.score(x_test, y_test))