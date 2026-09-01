import streamlit as st
import pandas as pd
import joblib

model = joblib.load('LogisticRegression.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')
