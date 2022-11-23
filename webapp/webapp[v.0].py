# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:42:42 2022

@author: samue
"""

import streamlit as st


st.title('Live Prediction')


p1 = st.slider('n_tokens_title', 0, 30 ,10, step = 1)

p2 = st.slider('n_tokens_content', 0.00, 1.00,0.5, step = 0.01)

p3 = st.slider('n_unique_tokens', 0.00, 1.00,0.5, step = 0.01)

p4 = st.slider('n_non_stop_words', 0.00, 1.00, step = 0.01)

p5 = st.slider('num_hrefs', 0.00, 1.00, step = 0.01)

p6 = st.slider('num_self_hrefs', 0.00, 1.00, step = 0.01)

p7 = st.slider('num_imgs', 0.00, 1.00, step = 0.01)

p8 = st.slider('num_videos', 0.00, 1.00, step = 0.01)

p9 = st.slider('average_token_length', 0.00, 1.00, step = 0.01)

p10 = st.slider('num_keywords', 0.00, 1.00, step = 0.01)

p11 = st.selectbox(
    'How would you like to be contacted?',
    ('lifestyle', 'entertainment', 'bus','socmed','tech','world','other'))
st.write('You selected:', p11)


p12 = st.slider('kw_min_min', 0.00, 1.00, step = 0.01)

p13 = st.slider('kw_min_max', 0.00, 1.00, step = 0.01)

p14 = st.slider('kw_max_max', 0.00, 1.00, step = 0.01)


p15 = st.slider('kw_avg_max', 0.00, 1.00, step = 0.01)


p16 = st.slider('kw_min_avg', 0.00, 1.00, step = 0.01)

p17 =st.slider('kw_max_avg', 0.00, 1.00, step = 0.01)


p18 = st.slider('self_reference_min_shares', 0.00, 1.00, step = 0.01)


p19 = st.slider('self_reference_max_shares', 0.00, 1.00, step = 0.01)


p20 = st.selectbox(
    'How would you like to be contacted?',
    ('monday', 'tuesday', 'wednesday','thursday','friday','saturday','sunday'))

st.write('You selected:', p20)

p21 = st.slider('LDA_00', 0.00, 1.00, step = 0.01)


p22 = st.slider('LDA_01', 0.00, 1.00, step = 0.01)

p23 = st.slider('LDA_02', 0.00, 1.00, step = 0.01)

p24 = st.slider('LDA_03', 0.00, 1.00, step = 0.01)

p25 = st.slider('LDA_04', 0.00, 1.00, step = 0.01)

p26 = st.slider('global_subjectivity', 0.00, 1.00, step = 0.01)

p27 = st.slider('global_sentiment_polarity', 0.00, 1.00, step = 0.01)

p28 = st.slider('global_rate_positive_words', 0.00, 0.20, step = 0.01)

p29 =  st.slider('global_rate_negative_words', 0.00, 0.20, step = 0.01)

p30 = st.slider('avg_positive_polarity', 0.00, 1.00, step = 0.01)

p31 =st.slider('min_positive_polarity', 0.00, 1.00, step = 0.01)

p32 = st.slider('max_positive_polarity', 0.00, 1.00, step = 0.01)

p33 = st.slider('avg_negative_polarity', -1.00, 0.00, step = 0.01)

p34 = st.slider('max_negative_polarity', -1.00, 0.00, step = 0.01)

p35 = st.slider('title_subjectivity', 0.00, 1.00, step = 0.01)

p36 =st.slider('title_sentiment_polarity', 0.00, 1.00, step = 0.01)

p37 = st.slider('abs_title_subjectivity', 0.00, 1.00, step = 0.01)


p38 = st.slider('NbVisit', 0.00, 1.00, step = 0.01)
st.write('The current movie title is', p38)

if st.button('Predict !'):
    st.write('j ai pas encore fini ')
