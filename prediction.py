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
