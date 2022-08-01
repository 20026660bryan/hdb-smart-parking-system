import pandas as pd
import numpy as np
import pickle
import streamlit as st
import re
from datetime import datetime
from datetime import timedelta
import random
import time
import requests
from streamlit_lottie import st_lottie


#--- LOAD MODELS ---
KNN = pickle.load(open('PKL_files/tunedKNN.pkl', 'rb'))
DecisionTree = pickle.load(open('PKL_files/tunedDecisionTree.pkl', 'rb'))
NaiveBayes = pickle.load(open('PKL_files/tunedNaiveBayes.pkl', 'rb'))
RandomForest = pickle.load(open('PKL_files/tunedrandomforest.pkl', 'rb'))
LogisticRegression = pickle.load(open('PKL_files/tunedlogisticregression.pkl', 'rb'))
SVM = pickle.load(open('PKL_files/tunedSVM.pkl', 'rb'))
MLP = pickle.load(open('PKL_files/tunedMLP.pkl', 'rb'))
XGBoost = pickle.load(open('PKL_files/tunedxgboost.pkl', 'rb'))


#--- DEFINE MODELS ---
def KNNparkingprediction(sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle):
    parkingpredictKNN = KNN.predict([[sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle]])
    return parkingpredictKNN

def dtparkingprediction(sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle):
    parkingpredictdt = DecisionTree.predict([[sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle]])
    return parkingpredictdt

def nbparkingprediction(sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle):
    parkingpredictnb = NaiveBayes.predict([[sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle]])
    return parkingpredictnb

def rfparkingprediction(sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle):
    parkingpredictrf = RandomForest.predict([[sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle]])
    return parkingpredictrf

def lrparkingprediction(sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle):
    parkingpredictlr = LogisticRegression.predict([[sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle]])
    return parkingpredictlr

def svmparkingprediction(sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle):
    parkingpredictsvm = SVM.predict([[sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle]])
    return parkingpredictsvm

def mlpparkingprediction(sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle):
    parkingpredictmlp = MLP.predict([[sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle]])
    return parkingpredictmlp

def xgbparkingprediction(sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle):
    parkingpredictxgb = XGBoost.predict([[sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle]])
    return parkingpredictxgb

#--- DEFINE LOTTIE URL---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#--- LOAD LOTTIE URL ---
lottie_coding = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_swnrn2oy.json")

#--- MAIN PAGE ---
def main():
    st.set_page_config(page_title="HDB Smart Parking System", page_icon=":red_car:", layout="wide")
    
    #--- Hide streamlit hamburger menu ---
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    #--- Sidebar containing algorithms ---
    with st.sidebar:
        with st.container():
            st.title('AI Algorithms')
            st.write('---')
            algorithms = ['Default(XGBoost)', 'K-Nearest Neighbours (KNN)', 'Decision Tree', 'Naive Bayes', 'Random Forest', 'Logistic Regression', 'SVM', 'MLP']
            model = st.selectbox('Select model', algorithms)
            
    #--- CONTAINER FOR INPUT ---
    with st.container():
        st.title('HDB Smart Parking System')
        st.write('---')
        left_column, right_column = st.columns(2)

    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")

    with left_column:
        st.subheader('Enter the following fields')
        license = st.text_input('License Plate Number').upper()
        
        #--- Defining checksum for license plate ---
        checksum = {0: 'A', 1: 'Z', 2: 'Y', 3: 'X', 4: 'U', 5: 'T', 6: 'S', 7: 'R', 8: 'P', 9: 'M', 10: 'L',
                11: 'K', 12: 'J', 13: 'H', 14: 'G', 15: 'E', 16: 'D', 17: 'C', 18: 'B'}

        checksumvalid = 0

        if license != '':
            numerical = list()
            prefix = ''

            for character in license:
                try:
                    numerical.append(int(character))
                except ValueError:
                    prefix += character

            numforprefix = []

            for z in prefix:
                numbers = ord(z) - 64
                numforprefix.append(numbers)

            prefix1 = 0
            prefix2 = 0
            prefix3 = 0

            if len(prefix) != 0:
                if len(numerical) != 0:
                    if len(numerical) == 4:
                        if len(prefix) == 4:
                            prefix1 = numforprefix[1] * 9
                            prefix2 = numforprefix[2] * 4

                        elif len(prefix) == 3:
                            prefix1 = numforprefix[0] * 9
                            prefix2 = numforprefix[1] * 4

                        num1 = numerical[0] * 5
                        num2 = numerical[1] * 4
                        num3 = numerical[2] * 3
                        num4 = numerical[3] * 2

                        sum = prefix1 + prefix2 + num1 + num2 + num3 + num4

                        mod = sum % 19

                        for a, b in checksum.items():
                            if mod == a:
                                if prefix[-1] == b:
                                    checksumvalid = 1

            if checksumvalid == 0 or len(prefix) != 4:
                errorplate = '<p style="font-family:sans-serif; color:Red; font-size: 12px;">Invalid license plate</p>'
                st.markdown(errorplate, unsafe_allow_html=True)
        
        #--- Defining regex patterns for motocycle, car and goods vehicle---
        motorcycle_pattern = 'FB[^IO][1-9][0-9]{0,3}[A-E | G-H | J-M | P | R-U | X-Z]'

        car_pattern = 'S[B-N][^IO][1-9][0-9]{0,3}[A-E | G-H | J-M | P | R-U | X-Z]'

        goods_pattern = 'GB[A-L][1-9][0-9]{0,3}[A-E | G-H | J-M | P | R-U | X-Z]'

        now = datetime.now()
        strnow = now.strftime('%H:%M ' + '| ' +  '%d' + 'th ' + '%B %Y')
        dtnow = datetime.strptime(strnow, '%H:%M ' + '| ' +  '%d' + 'th ' + '%B %Y')
        nowhoursandminutes = dtnow.strftime('%H:%M')

        dur = st.number_input('Duration (in minutes)', min_value = 0, step = 30)
        
        validdur = 0
        
        #--- Validation check for duration (No need to check < 0 because user cannot input below 0) ---
        if dur > 999999:
             errordur = '<p style="font-family:sans-serif; color:Red; font-size: 12px;">Invalid duration</p>'
             st.markdown(errordur, unsafe_allow_html=True)
             validdur = 0
        elif dur > 0 and dur <= 999999:
            validdur = 1

        #else:
        end = dtnow + timedelta(minutes = dur)
        total_hours = dur // 60
        total_mins = dur % 60
        
        if dur == 0:
            blank = ''
            st.markdown(blank, unsafe_allow_html=True)
            validdur = 0
        else:
            st.code('{} hours {} minutes'.format(total_hours, total_mins))
        # st.code('{} hours {} minutes'.format(total_hours, total_mins),language="python")
        
        #--- Method to print parking expiry date based on duration input ---
        strendth = end.strftime('%H:%M ' + '| ' +  '%d' + 'th ' + '%B %Y')
        strendst = end.strftime('%H:%M ' + '| ' +  '%d' + 'st ' + '%B %Y')
        strendnd = end.strftime('%H:%M ' + '| ' +  '%d' + 'nd ' + '%B %Y')
        strendrd = end.strftime('%H:%M ' + '| ' +  '%d' + 'rd ' + '%B %Y')
        dtend = datetime.strptime(strendth, '%H:%M ' + '| ' + '%d' + 'th ' + '%B %Y')

        if end.day == 1 or end.day == 21 or end.day == 31:

            title = '<p style="font-family:sans-serif; color:Black; font-size: 12px;"> Parking Expiry Date</p>'

            st.markdown(title, unsafe_allow_html=True)
            st.info(strendst)

        elif end.day == 2 or end.day == 22:

            title = '<p style="font-family:sans-serif; color:Black; font-size: 12px;"> Parking Expiry Date</p>'

            st.markdown(title, unsafe_allow_html=True)
            st.info(strendnd)

        elif end.day == 3 or end.day == 23:

            title = '<p style="font-family:sans-serif; color:Black; font-size: 12px;"> Parking Expiry Date</p>'

            st.markdown(title, unsafe_allow_html=True)
            st.info(strendrd)

        else:

            st.info(strendth)

        epochstart = dtnow.timestamp() / 60
        epochend = dtend.timestamp() / 60

        sessionstart = epochstart

        sessionend = epochend

        totalcharge = 0
        vehicle = 0
        valid = 0
        
        #--- Method to check license plate with regex ---
        if re.search(motorcycle_pattern, license):
            if checksumvalid == 1:
                totalcharge = 0.65
                vehicle = 1
                valid = 1
        elif re.search(car_pattern, license):
            if checksumvalid == 1:
                if dur > 1440:
                    totalcharge = 0
                else:
                    totalcharge = (dur * 0.02)
                vehicle = 0
                valid = 1
        elif re.search(goods_pattern, license):
            if checksumvalid == 1:
                if dur > 1440:
                    totalcharge = 0
                else:
                    totalcharge = (dur * 0.04)
                vehicle = 2
                valid = 1
        else:
            valid = 0



    effectivecharge = totalcharge

    predict = ''
    parkingtype = ''
    
    #--- CONTAINER FOR OUTPUT---
    with st.container():
        left_column, right_column = st.columns(2)
        with left_column:
            st.write('---')
            if checksumvalid == 1:
                if valid == 1 and validdur == 1:
                    if st.button('Predict', disabled = False):
                        with st.spinner('Give us a moment...'):
                            time.sleep(2)
                            
                            #--- XGBOOST ---
                        if model == 'Default(XGBoost)':
                            predict = xgbparkingprediction(sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle)
                            if vehicle == 0:
                                if predict == 0:
                                    parkingtype = 'Short Term Parking'
                                elif predict == 1:
                                    parkingtype = 'Seasonal Parking'
                            elif vehicle == 1:
                                        parkingtype = 'Motorcycle Parking'
                            elif vehicle == 2:
                                if predict == 0:
                                    parkingtype = 'Short Term Parking'
                                elif predict == 1:
                                    parkingtype = 'Seasonal Parking'

                            st.write("---")
                            st.subheader('License Plate: ' + license)
                            # st.write('Parking Type: ' + parkingtype)

                            st.success(parkingtype)

                            seasonal = random.randint(1, 160)
                            shortterm = random.randint(161, 480)
                            motorcycle = random.randint(481, 500)

                            if vehicle == 0:
                                if parkingtype == 'Seasonal Parking':
                                    st.success('Lot Number: ' + str(seasonal))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                                elif parkingtype == 'Short Term Parking':
                                    st.success('Lot Number: ' + str(shortterm))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                            elif vehicle == 1:

                                st.success('Lot Number: ' + str(motorcycle))
                                st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                            elif vehicle == 2:
                                if parkingtype == 'Seasonal Parking':
                                    st.success('Lot Number: ' + str(seasonal))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                                elif parkingtype == 'Short Term Parking':

                                    st.success('Lot Number: ' + str(shortterm))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")
                                    
                        #--- KNN ---
                        elif model == 'K-Nearest Neighbours (KNN)':
                            predict = KNNparkingprediction(sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle)
                            if vehicle == 0:
                                if predict == 0:
                                    parkingtype = 'Short Term Parking'
                                elif predict == 1:
                                    parkingtype = 'Seasonal Parking'
                            elif vehicle == 1:
                                        parkingtype = 'Motorcycle Parking'
                            elif vehicle == 2:
                                if predict == 0:
                                    parkingtype = 'Short Term Parking'
                                elif predict == 1:
                                    parkingtype = 'Seasonal Parking'

                            st.write("---")
                            st.subheader('License Plate: ' + license)
                            # st.write('Parking Type: ' + parkingtype)

                            st.success(parkingtype)

                            seasonal = random.randint(1, 160)
                            shortterm = random.randint(161, 480)
                            motorcycle = random.randint(481, 500)

                            if vehicle == 0:
                                if parkingtype == 'Seasonal Parking':
                                    st.success('Lot Number: ' + str(seasonal))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                                elif parkingtype == 'Short Term Parking':
                                    st.success('Lot Number: ' + str(shortterm))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                            elif vehicle == 1:

                                st.success('Lot Number: ' + str(motorcycle))
                                st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                            elif vehicle == 2:
                                if parkingtype == 'Seasonal Parking':
                                    st.success('Lot Number: ' + str(seasonal))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                                elif parkingtype == 'Short Term Parking':

                                    st.success('Lot Number: ' + str(shortterm))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                        #--- DECISION TREE ---
                        elif model == 'Decision Tree':
                            predict = dtparkingprediction(sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle)
                            if vehicle == 0:
                                if predict == 0:
                                    parkingtype = 'Short Term Parking'
                                elif predict == 1:
                                    parkingtype = 'Seasonal Parking'
                            elif vehicle == 1:
                                        parkingtype = 'Motorcycle Parking'
                            elif vehicle == 2:
                                if predict == 0:
                                    parkingtype = 'Short Term Parking'
                                elif predict == 1:
                                    parkingtype = 'Seasonal Parking'

                            st.write("---")
                            st.subheader('License Plate: ' + license)
                            # st.write('Parking Type: ' + parkingtype)

                            st.success(parkingtype)

                            seasonal = random.randint(1, 160)
                            shortterm = random.randint(161, 480)
                            motorcycle = random.randint(481, 500)

                            if vehicle == 0:
                                if parkingtype == 'Seasonal Parking':
                                    st.success('Lot Number: ' + str(seasonal))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                                elif parkingtype == 'Short Term Parking':
                                    st.success('Lot Number: ' + str(shortterm))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                            elif vehicle == 1:

                                st.success('Lot Number: ' + str(motorcycle))
                                st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                            elif vehicle == 2:
                                if parkingtype == 'Seasonal Parking':
                                    st.success('Lot Number: ' + str(seasonal))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                                elif parkingtype == 'Short Term Parking':

                                    st.success('Lot Number: ' + str(shortterm))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")
 
                        #--- NAIVE BAYES ---
                        elif model == 'Naive Bayes':
                            predict = nbparkingprediction(sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle)
                            if vehicle == 0:
                                if predict == 0:
                                    parkingtype = 'Short Term Parking'
                                elif predict == 1:
                                    parkingtype = 'Seasonal Parking'
                            elif vehicle == 1:
                                        parkingtype = 'Motorcycle Parking'
                            elif vehicle == 2:
                                if predict == 0:
                                    parkingtype = 'Short Term Parking'
                                elif predict == 1:
                                    parkingtype = 'Seasonal Parking'

                            st.write("---")
                            st.subheader('License Plate: ' + license)
                            # st.write('Parking Type: ' + parkingtype)

                            st.success(parkingtype)

                            seasonal = random.randint(1, 160)
                            shortterm = random.randint(161, 480)
                            motorcycle = random.randint(481, 500)

                            if vehicle == 0:
                                if parkingtype == 'Seasonal Parking':
                                    st.success('Lot Number: ' + str(seasonal))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                                elif parkingtype == 'Short Term Parking':
                                    st.success('Lot Number: ' + str(shortterm))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                            elif vehicle == 1:

                                st.success('Lot Number: ' + str(motorcycle))
                                st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                            elif vehicle == 2:
                                if parkingtype == 'Seasonal Parking':
                                    st.success('Lot Number: ' + str(seasonal))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                                elif parkingtype == 'Short Term Parking':

                                    st.success('Lot Number: ' + str(shortterm))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")
                        
                        #--- RANDOM FOREST ---
                        elif model == 'Random Forest':
                            predict = rfparkingprediction(sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle)
                            if vehicle == 0:
                                if predict == 0:
                                    parkingtype = 'Short Term Parking'
                                elif predict == 1:
                                    parkingtype = 'Seasonal Parking'
                            elif vehicle == 1:
                                        parkingtype = 'Motorcycle Parking'
                            elif vehicle == 2:
                                if predict == 0:
                                    parkingtype = 'Short Term Parking'
                                elif predict == 1:
                                    parkingtype = 'Seasonal Parking'

                            st.write("---")
                            st.subheader('License Plate: ' + license)
                            # st.write('Parking Type: ' + parkingtype)

                            st.success(parkingtype)

                            seasonal = random.randint(1, 160)
                            shortterm = random.randint(161, 480)
                            motorcycle = random.randint(481, 500)

                            if vehicle == 0:
                                if parkingtype == 'Seasonal Parking':
                                    st.success('Lot Number: ' + str(seasonal))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                                elif parkingtype == 'Short Term Parking':
                                    st.success('Lot Number: ' + str(shortterm))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                            elif vehicle == 1:

                                st.success('Lot Number: ' + str(motorcycle))
                                st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                            elif vehicle == 2:
                                if parkingtype == 'Seasonal Parking':
                                    st.success('Lot Number: ' + str(seasonal))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                                elif parkingtype == 'Short Term Parking':

                                    st.success('Lot Number: ' + str(shortterm))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")
                        
                        #--- LOGISTIC REGRESSION ---
                        elif model == 'Logistic Regression':
                            predict = lrparkingprediction(sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle)
                            if vehicle == 0:
                                if predict == 0:
                                    parkingtype = 'Short Term Parking'
                                elif predict == 1:
                                    parkingtype = 'Seasonal Parking'
                            elif vehicle == 1:
                                        parkingtype = 'Motorcycle Parking'
                            elif vehicle == 2:
                                if predict == 0:
                                    parkingtype = 'Short Term Parking'
                                elif predict == 1:
                                    parkingtype = 'Seasonal Parking'

                            st.write("---")
                            st.subheader('License Plate: ' + license)
                            # st.write('Parking Type: ' + parkingtype)

                            st.success(parkingtype)

                            seasonal = random.randint(1, 160)
                            shortterm = random.randint(161, 480)
                            motorcycle = random.randint(481, 500)

                            if vehicle == 0:
                                if parkingtype == 'Seasonal Parking':
                                    st.success('Lot Number: ' + str(seasonal))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                                elif parkingtype == 'Short Term Parking':
                                    st.success('Lot Number: ' + str(shortterm))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                            elif vehicle == 1:

                                st.success('Lot Number: ' + str(motorcycle))
                                st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                            elif vehicle == 2:
                                if parkingtype == 'Seasonal Parking':
                                    st.success('Lot Number: ' + str(seasonal))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                                elif parkingtype == 'Short Term Parking':

                                    st.success('Lot Number: ' + str(shortterm))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")
                                    
                        #--- SVM ---
                        elif model == 'SVM':
                            predict = svmparkingprediction(sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle)
                            if vehicle == 0:
                                if predict == 0:
                                    parkingtype = 'Short Term Parking'
                                elif predict == 1:
                                    parkingtype = 'Seasonal Parking'
                            elif vehicle == 1:
                                        parkingtype = 'Motorcycle Parking'
                            elif vehicle == 2:
                                if predict == 0:
                                    parkingtype = 'Short Term Parking'
                                elif predict == 1:
                                    parkingtype = 'Seasonal Parking'

                            st.write("---")
                            st.subheader('License Plate: ' + license)
                            # st.write('Parking Type: ' + parkingtype)

                            st.success(parkingtype)

                            seasonal = random.randint(1, 160)
                            shortterm = random.randint(161, 480)
                            motorcycle = random.randint(481, 500)

                            if vehicle == 0:
                                if parkingtype == 'Seasonal Parking':
                                    st.success('Lot Number: ' + str(seasonal))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                                elif parkingtype == 'Short Term Parking':
                                    st.success('Lot Number: ' + str(shortterm))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                            elif vehicle == 1:

                                st.success('Lot Number: ' + str(motorcycle))
                                st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                            elif vehicle == 2:
                                if parkingtype == 'Seasonal Parking':
                                    st.success('Lot Number: ' + str(seasonal))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                                elif parkingtype == 'Short Term Parking':

                                    st.success('Lot Number: ' + str(shortterm))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")
                        
                        #--- MLP ---
                        elif model == 'MLP':
                            predict = mlpparkingprediction(sessionstart, sessionend, totalcharge, dur, effectivecharge, vehicle)
                            if vehicle == 0:
                                if predict == 0:
                                    parkingtype = 'Short Term Parking'
                                elif predict == 1:
                                    parkingtype = 'Seasonal Parking'
                            elif vehicle == 1:
                                        parkingtype = 'Motorcycle Parking'
                            elif vehicle == 2:
                                if predict == 0:
                                    parkingtype = 'Short Term Parking'
                                elif predict == 1:
                                    parkingtype = 'Seasonal Parking'

                            st.write("---")
                            st.subheader('License Plate: ' + license)
                            # st.write('Parking Type: ' + parkingtype)

                            st.success(parkingtype)

                            seasonal = random.randint(1, 160)
                            shortterm = random.randint(161, 480)
                            motorcycle = random.randint(481, 500)

                            if vehicle == 0:
                                if parkingtype == 'Seasonal Parking':
                                    st.success('Lot Number: ' + str(seasonal))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                                elif parkingtype == 'Short Term Parking':
                                    st.success('Lot Number: ' + str(shortterm))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                            elif vehicle == 1:

                                st.success('Lot Number: ' + str(motorcycle))
                                st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                            elif vehicle == 2:
                                if parkingtype == 'Seasonal Parking':
                                    st.success('Lot Number: ' + str(seasonal))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")

                                elif parkingtype == 'Short Term Parking':

                                    st.success('Lot Number: ' + str(shortterm))
                                    st.success('Parking fees: ' + str('$' + '{:.2f}'.format(totalcharge)))
                                    st.write("[(Learn more about parking rates)](https://www.hdb.gov.sg/car-parks/shortterm-parking/short-term-parking-charges)")


            else:
                st.button('Predict', disabled = True)

if __name__ == '__main__':
    main()
