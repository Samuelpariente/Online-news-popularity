# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 17:34:39 2022

@author: samue
"""

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import bar_chart_race as bcr
import base64

v_news = pd.read_csv('Webapp/v_news.csv')
i = 0
def couleur(*args, **kwargs):
        global i 
        if i < 11:
            i = i+1
            return "rgb(255, 0, 0)" 
        if i > 10:
            return "rgb(0, 0, 0)"

def main_page():
    st.markdown("# Overview ")
    st.sidebar.markdown("# Overview")
    """
    ## Dataset Description
    This dataset summarizes a heterogeneous set of features about articles published by Mashable in a period of two years. 
    The articles were published by Mashable (www.mashable.com) and their content as the rights to reproduce it belongs to them. Hence, this dataset does not share the original content but some statistics associated with it. Acquisition date: January 8, 2015.
    
    We scapped with selenium from mashable Authors and titles: 
    
    """
    news = pd.read_csv('OnlineNewsPopularityWithAutorsAndTitles.csv')
    news[['Autors','titles']]
    """
    To that we add the number of visits by day with google trends:
    """
    timeline = pd.read_csv('multiTimeline.csv')
    timeline
    
    """
    Moreover we had 2 qualitatives columns: Week days and Types of news. We decided to let week day in  One hot encoding and to vectorise chanel thank to a Google news vectorisation model. 
    
    """
    image = Image.open('illustration1.PNG')
    st.image(image, caption='Vectorisation')
    
    """
    We finished with 3 predictable Variables:
    """

    fig = px.histogram(v_news, x="Class_shares1")
    st.plotly_chart(fig)

    fig = px.histogram(v_news, x="Class_shares2")
    st.plotly_chart(fig)
    

    fig = px.histogram(v_news, x="shares")
    st.plotly_chart(fig)
    
    
    
    
def page2():
    st.markdown("# Data discovery ")
    st.sidebar.markdown("Data discovery ")
    
    
    
    chanel = v_news.groupby(by="Chanel").Chanel.count()
    
    fig = go.Figure(data=go.Scatterpolar(
      r=chanel.values,
      theta=chanel.index,
      fill='toself',
      name='Frequencies of Chanels'
    ))
    
    fig.update_layout(
      title={
          'text': "Frequencies of Chanels", 
          'y': 0.9,
          'x': 0.5
      },
      polar=dict(
        radialaxis=dict(
          visible=True
        ),
      ),
      showlegend=False
    )
    st.plotly_chart(fig)
       
    autnb = v_news.groupby('Authors').count()['Titles'].sort_values(ascending = False)[:10]
    autnb =autnb.reset_index()
    
    
    fig = px.sunburst(autnb, path=['Authors'], values = 'Titles',title = "TOP 10 des acteurs ayant joué dans le plus de genres différents")
    st.plotly_chart(fig)
    words = []
    
    for x in v_news['Titles']:
        for y in x.split(' '):
            words.append(y)
    text = ' '.join(words)
    mask = np.array(Image.open('Mlogo.png'))
    mask[mask == 0] = 255
    wordcloud = WordCloud(background_color = 'white', max_words = 200, mask =mask,contour_width=1).generate(text)
    
    
    
    plt.subplots(figsize=(60, 20))
    plt.imshow(wordcloud.recolor(color_func = couleur))
    plt.imshow(wordcloud)
    plt.axis("off")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown("## Most Frequent words ")
    st.pyplot()
    
    v_news.hist(figsize=(20,20))
    
    
    st.markdown("## Univariate analisis of quantitative variables ")
    st.pyplot()
    
    lda = v_news.loc[:, "LDA_00":"LDA_04"]
    pca = PCA(n_components=3)
    pca.fit(lda)
    explanation_coefs = pca.explained_variance_ratio_
    pca_values = pca.transform(lda)
    np.sum(explanation_coefs)
    lda["Chanel"] = v_news["Chanel"]
    map_colors = {
        "Entertainment":"limegreen",
        "Business": "darkturquoise",
        "Tech": "orchid",
        "LifeStyle": "royalblue",
        "World": "gold",
        "Others": "tomato",
        "Social Media": "pink"
    }
    colors = v_news["Chanel"].map(map_colors)
    
    fig = px.scatter_3d(x=pca_values[:200, 0], y=pca_values[:200, 1], z=pca_values[:200, 2],
                  color = lda["Chanel"].head(200), opacity=0.7)
            
    # tight layout
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        legend_title="Chanels"
    )
    
    st.markdown("## Proximity of chanels")
    st.plotly_chart(fig)
    

    
    
    top_10_authors = v_news.groupby(by="Authors").shares.sum().sort_values(ascending=False).head(10).index.values
    
    top_10_authors_race = v_news[v_news["Authors"].isin(top_10_authors)].reset_index(drop=True)[["Authors", "shares", "date"]]
    temp_date = pd.to_datetime(top_10_authors_race["date"]).dt.strftime('%m/%Y')
    top_10_authors_race["date"] = pd.to_datetime(temp_date)
    
    table_top_10_authors_race = pd.pivot_table(top_10_authors_race, 
                                               index="Authors", 
                                               columns="date", 
                                               values="shares", 
                                               aggfunc="sum").fillna(0).cumsum(axis=1)
    table_top_10_authors_race = table_top_10_authors_race.T 
    
    html_str = bcr.bar_chart_race(
        df=table_top_10_authors_race,
        #filename='covid19_horiz.mp4',
        orientation='h',
        sort='desc',
        n_bars=10,
        fixed_order=False,
        fixed_max=True,
        steps_per_period=2,
        interpolate_period=False,
        label_bars=True,
        bar_size=.95,
        period_label={'x': .99, 'y': .25, 'ha': 'right', 'va': 'center'},
        period_fmt='%Y',
        period_summary_func=lambda v, r: {'x': .99, 'y': .18,
                                          's': f'Nombre total de partages: {v.nlargest(6).sum():,.0f}',
                                          'ha': 'right', 'size': 8, 'family': 'Courier New'},
        perpendicular_bar_func='median',
        period_length=150,
        figsize=(5, 3),
        dpi=144,
        cmap='dark12',
        title='TOP 10 most influencial authors of Mashable over time',
        title_size='',
        bar_label_size=7,
        tick_label_size=7,
        writer=None,
        fig=None,
        bar_kwargs={'alpha': .7},
        filter_column_colors=False).data
    
    start = html_str.find('base64,')+len('base64,')
    end = html_str.find('">')
    
    video = base64.b64decode(html_str[start:end])
    st.video(video)
    
    corr = v_news.corr()
    fig = px.imshow(corr,color_continuous_scale='RdBu_r', text_auto=True)
    st.plotly_chart(fig)
    
def page3():
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")

def page4():
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
    

page_names_to_funcs = {
    "Overview": main_page,
    "Data preprocess/discovery": page2,
    "Improve your article": page3,
    "Predict your success": page4,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
