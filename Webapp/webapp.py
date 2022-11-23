# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:42:42 2022

@author: samue
"""

import streamlit as st
import requests

st.title('Live Prediction')


p1 = st.slider('n_tokens_title', 0, 30 ,10, step = 1)

p2 = st.slider('n_tokens_content', 0, 10000,564, step = 1)

p3 = st.slider('n_unique_tokens', 0.00, 1.00,0.54, step = 0.01)

p4 = st.slider('n_non_stop_words', 0.00, 1.00,1.00, step = 0.01)

p5 = st.slider('num_hrefs', 0, 350,11, step = 1)

p6 = st.slider('num_self_hrefs', 0, 150,3, step = 1)

p7 = st.slider('num_imgs', 0, 100,5, step = 1)

p8 = st.slider('num_videos', 0, 150,1, step = 1)

p9 = st.slider('average_token_length', 0.00, 10.00,4.6, step = 0.01)

p10 = st.slider('num_keywords', 0.00, 10.00,7.2, step = 0.01)

p11 = st.selectbox(
    'How would you like to be contacted?',
    ('lifestyle', 'entertainment', 'bus','socmed','tech','world'))
st.write('You selected:', p11)


p12 = st.slider('kw_min_min', 0, 300000,1150, step = 1)

p122 = st.slider('kw_max_min', 0, 300000,1152, step = 1)

p13 = st.slider('kw_min_max', 0, 843300,13062, step = 1)

p14 = st.slider('kw_max_max', 0, 843300,750683, step = 1)


p15 = st.slider('kw_avg_max', 0,843300,255064, step = 1)


p16 = st.slider('kw_min_avg', 0, 3613, 1094,step = 1)

p17 =st.slider('kw_max_avg', 0, 298400,5582, step = 1)


p18 = st.slider('self_reference_min_shares', 0, 850000,4131, step = 1)


p19 = st.slider('self_reference_max_shares', 0, 850000,10585, step = 1)


p20 = st.selectbox(
    'How would you like to be contacted?',
    ('monday', 'tuesday', 'wednesday','thursday','friday','saturday','sunday'))

st.write('You selected:', p20)

p21 = st.slider('LDA_00', 0.00, 1.00,0.18, step = 0.01)


p22 = st.slider('LDA_01', 0.00, 1.00,0.14, step = 0.01)

p23 = st.slider('LDA_02', 0.00, 1.00,0.21, step = 0.01)

p24 = st.slider('LDA_03', 0.00, 1.00,0.21, step = 0.01)

p25 = st.slider('LDA_04', 0.00, 1.00,0.23, step = 0.01)

p26 = st.slider('global_subjectivity', 0.00, 1.00,0.45, step = 0.01)

p27 = st.slider('global_sentiment_polarity',0.00, 1.00,0.12, step = 0.01)

p28 = st.slider('global_rate_positive_words', 0.00, 0.20,0.04, step = 0.01)

p29 =  st.slider('global_rate_negative_words,', 0.00, 0.20,0.01, step = 0.01)

p30 = st.slider('avg_positive_polarity', 0.00, 1.00,0.36, step = 0.01)

p31 =st.slider('min_positive_polarity', 0.00, 1.00,0.09, step = 0.01)

p32 = st.slider('max_positive_polarity', 0.00, 1.00,0.77, step = 0.01)

p33 = st.slider('avg_negative_polarity', -1.00, 0.00,-0.26, step = 0.01)

p34 = st.slider('max_negative_polarity', -1.00, 0.00,-0.11, step = 0.01)

p35 = st.slider('title_subjectivity', 0.00, 1.00,0.28, step = 0.01)

p36 =st.slider('title_sentiment_polarity', 0.00, 1.00,0.06, step = 0.01)

p37 = st.slider('abs_title_subjectivity', 0.00, 1.00,0.34, step = 0.01)

p38 = st.slider('NbVisit', 0.00, 15.00,7.41,step = 0.01)

if p20 == 'monday':
    p20 = '1/0/0/0/0/0/0'

if p20 == 'tuesday':
    p20 = '0/1/0/0/0/0/0'

if p20 == 'wednesday':
    p20 = '0/0/1/0/0/0/0'

if p20 == 'thursday':
    p20 = '0/0/0/1/0/0/0'

if p20 == 'friday':
    p20 = '0/0/0/0/1/0/0'

if p20 == 'saturday':
    p20 = '0/0/0/0/0/1/0'

if p20 == 'sunday':
    p20 = '0/0/0/0/0/0/1'


if p11 == 'lifestyle':
    p11 = '1/0/0/0/0/0'

if p11 == 'entertainment':
    p11 = '0/1/0/0/0/0'

if p11 == 'bus':
    p11 = '0/0/1/0/0/0'

if p11 == 'socmed':
    p11 = '0/0/0/1/0/0'

if p11 == 'tech':
    p11 = '0/0/0/0/1/0'

if p11 == 'world':
    p11 = '0/0/0/0/0/1'



info1 = str(p1)+'/'+str(p2)+'/'+str(p3)+'/'+str(p4)+'/'+str(p5)+'/'+str(p6)+'/'+str(p7)+'/'+str(p8)+'/'+str(p9)+'/'+str(p10)+'/'+str(p11)+'/'+str(p12)+'/'+str(p122)+'/'+str(p13)+'/'+str(p14)+'/'+str(p15)+'/'+str(p16)+'/'+str(p17)+'/'+str(p18)+'/'+str(p19)+'/'+str(p20)+'/'+str(p21)+'/'+str(p22)+'/'+str(p23)+'/'+str(p24)+'/'+str(p25)+'/'+str(p26)+'/'+str(p27)+'/'+str(p28)+'/'+str(p29)+'/'+str(p30)+'/'+str(p31)+'/'+str(p32)+'/'+str(p33)+'/'+str(p34)+'/'+str(p35)+'/'+str(p36)+'/'+str(p37)+'/'+str(p38)
info = 'http://3.80.212.138:8080/predict?info='+info1

if st.button('Predict !'):
    a = requests.get(info)
    st.write('Your article gona whave '+a.text+' shares' )