import sklearn
import joblib
import streamlit as st
import xgboost

st.title("Insurance prediction")

def get_region(reg):
    if reg == 'northwest':
        return [1,0,0]
    elif reg == 'southwest':
        return [0,0,1]
    elif reg == 'northeast':
        return [0,0,0]
    else:
        return [0,1,0]




age = st.number_input('Age',0,50)
bmi = st.number_input('BMI')
childern = st.number_input('children')
sex = st.radio("Gender",['male','female'])
smoker = st.checkbox('smoker')
region = st.selectbox('region',['northwest', 'southwest','northeast','southeast'])

if sex == 'male':
    sex = 1
else:
    sex = 0


row_data = [age, bmi, childern, sex, smoker]
row_data.extend(get_region(region))
model = joblib.load("model.h5")
scaler = joblib.load("scaler.h5")
predicted_value = round(model.predict(scaler.transform([row_data]))[0],2)


button = st.button('predict')
if button:
    st.markdown(f'{predicted_value}')
    

    