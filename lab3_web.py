from imp import load_module
import streamlit as st
import pandas as pd
import numpy as np

model = load_module('D:/HocMayVaUngDung/TH/LAB3_dataset/lab3/lab3')
st.write('ket qua: ',model)