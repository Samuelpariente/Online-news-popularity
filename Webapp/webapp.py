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
import seaborn as sns
from bokeh.palettes import Magma7
from bokeh.palettes import Pastel2_7
from bokeh.palettes import YlOrRd9
from bokeh.palettes import Greens9
from bokeh.palettes import Purples9

online = True

if online== True: 
    illustration1 = 'Webapp/illustration1.PNG'
    v_news = 'Webapp/v_news.csv'
    multiTimeline='Webapp/multiTimeline.csv'
    Mlogo = 'Webapp/Mlogo.png'
    OnlineNewsPopularityWithAutorsAndTitles= 'Webapp/OnlineNewsPopularityWithAutorsAndTitles.csv'
    
if online  == False: 
    illustration1 = 'illustration1.PNG'
    v_news = 'v_news.csv'
    multiTimeline='multiTimeline.csv'
    Mlogo = 'Mlogo.png'
    OnlineNewsPopularityWithAutorsAndTitles= 'OnlineNewsPopularityWithAutorsAndTitles.csv'

v_news = pd.read_csv(v_news)
i = 0
def couleur(*args, **kwargs):
        global i 
        if i < 11:
            i = i+1
            return "rgb(255, 0, 0)" 
        if i > 10:
            return "rgb(0, 0, 0)"
def library():
    st.markdown("# Libraries ")
    """
    In this section we load all the libraries we need for the notebook to be working properly. Here is the extensive list of utilized libraries :

    - **Data Analysis libraries** :
      - Datetime
      - Numpy
      - Pandas 
      - Missingno
    
    - **Visualization libraries** :
      - Seaborn
      - Matplotlib
      - Bokeh
      - Plotly
      - Wordcloud 
      - BarChartRace
    
    - **Machine and Deep Learning libraries** :
      - Sklearn 
      - Auto-Sklearn
      - Tensorflow
    
    - **Scrapping libraries** :
      - Selenium
    
    - **Api libraries**
      - Flask
      - Pickle
    
    - **Date Storage related libraries** :
      - Google Colab to Google Drive linkage 
    """
    
def main_page():
    st.markdown("# Overview ")
    st.sidebar.markdown("# Overview")
    st.markdown("## Datasets ")
    """
    In the context of our project we studied the online-news-popularity dataset of mashable articles. Our project contains 2 csv datasets loaded in this section :

    - **Online News Popularity Dataset**
      - 63 columns | 39644 rows
      - Base dataset 
    """
    news = pd.read_csv(OnlineNewsPopularityWithAutorsAndTitles)
    news
    """
    
    - **Timeline Dataset**
      - 2 columns | 103 rows
      - Contains "Mashable.com" internet user's flow for 103 dates 
    
    """

    timeline = pd.read_csv(multiTimeline)
    timeline
    
    st.markdown("## Data Pre-processing ")
    """
    In the prepocessing section, our objective in to make the data usable for visualisation and modeling usage.To do so, we have to perform multiple modifications on the dataset. Find the complete explanation below.

    ****
    
    1. **Scrapping and Addition of columns**
      - Using Selenium we add the title of each article and its author.
      - We also convert the "timedelta" to the actual release date of the article and put it in a new column.
      - We merge "timeline" dataset and "news" dataset in a new column.
      - We create a discretized version of the target column (shares) : a binary discretization and a multi class one.
    """
    code = """ 
urls = news["url"]
driver = webdriver.Chrome('chromedriver.exe')

driver.get(urls[0])
time.sleep(2)
element = driver.find_elements(by = By.CSS_SELECTOR, value = 'div[id="onetrust-button-group"]')
element[0].click()
authors = []
titles = []

authors = []
titles = []
for i in range(0, len(urls[i])):
    driver.get(urls[i])
    time.sleep(0.5)
    try : 
        timer1 = WebDriverWait(driver, 2).until(EC.presence_of_element_located((By.CSS_SELECTOR,'a[class="underline-link"]')))
        timer2 = WebDriverWait(driver, 2).until(EC.presence_of_element_located((By.CSS_SELECTOR,'h1[class="mt-4 header-100 max-w-5xl "]')))
        name = driver.find_elements(by = By.CSS_SELECTOR, value = 'a[class="underline-link"]')
        title = driver.find_elements(by = By.CSS_SELECTOR, value = 'h1[class="mt-4 header-100 max-w-5xl "]')

        authors.append(name[0].text)
        titles.append(title[0].text)

    except (TimeoutException , NameError):
        authors.append('Nan')
        titles.append('Nan')
news['Autors'] = authors
news['titles'] = titles
df.to_csv('OnlineNewsPopularityWithAutorsAndTitles.csv')
    """
    st.code(code, language="python")
    
    
    """
    ****
    
    2. **Cleaning**
      - We perform a "Na" values study and notice that our dataset doesn't contains any trivial "na" values.
      - *Columns 2 to 6 (Tokens related)* : we verify that ratios columns are between 0 and 1. We check as well if articles have a content or not.
      - *Colummns 13 to 18 (Chanels)* : In anticipation of the vectorization we decode the 1 hot encoding and create a new qualitive column.
      - *Columns 19 to 27 (Keywords)* : Given that Keywords related columns represent numbers of shares, we verify their non-negativity.
      - *Columns 31 to 37 (Day of the week)* : In a similary fashion as chanels columns, we decode the 1 hot encoding to make it a new qualitative column.
      - *Columns 39 to 43 (LDA topics)* : These columns are ratio. Thus, we verify that they all lie between 0 and 1.
      - *Columns 61 to 62 (Title and Author)* : We must verify the completeness of these scrapped columns. In fact we find out that 619 columns are "Nan". We suppress them because it is only a small proportion of the dataset.
    
    ***
    
    3. **Outlier Handling**
      - We decide to select the columns of the dataset for which "shares" columns lie between first and third quartiles of the dataset.
    
    """
    col1, col2 = st.columns(2)
    col1.metric(label="Dropped outliers", value="4457")
    col2.metric(label="news rows", value="33386", delta = "-11.7%")
    
    """
    ***
    
    4. **Vectorization**
      - First, we load a google vectorization model pre-trained on online articles.
      - Then we vectorize our chanel column using this model.
      - However, the output of the model is a 300 dimensional vector which perturbates learning of models, especially deep learning sequential ones. Thus, we compute a PCA on this output vector and transform it to a 5 dimensional vector that we can concatenate with our pre-existing dataset.
    """
    image = Image.open(illustration1)
    st.image(image, caption='Vectorisation')
    
    """
    ***
    
    5. **Creation of the working dataframes**
      - We create 2 dataframes to work with. 
      - The first one is the visualization dataframe, "v_news". It doesn't contain 1 hot encoded variables, or vectorized ones. 
      - The second one, named "m_news" and created for modeling purposes, has only numerical columns, encompassing vectorized and 1 hot encoded columns. In addition, we remove non predictive columns from it, such as timedelta. Finally, we only keep columns that are less than 70% percent correlated with others.

    
    
    6. **3 predictable Variables**
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
    mask = np.array(Image.open(Mlogo))
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
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown("## 3 - Recipe for the perfect article ")
    st.sidebar.markdown("## 3 - Recipe for the perfect article \n In this section we gonna study the best parameters to improve your article ")
    
    
    pol_subj_wr_shares = v_news[["global_subjectivity", "avg_positive_polarity", "avg_negative_polarity", "shares"]]

    discrete_pol_subj_wr_shares = pd.DataFrame()
    bins_v = [round(x, 1) for x in np.arange(0, 1.1, 0.1)]
    labels_v = [round(i, 1) for i in np.arange(0, 1, 0.1)]
    bins_v_neg = [round(x, 1) for x in np.arange(-1, 0.1, 0.1)]
    labels_v_neg = [round(i, 1) for i in np.arange(-1, 0, 0.1)]
    
    discrete_pol_subj_wr_shares["global_subjectivity"] = pd.cut(pol_subj_wr_shares["global_subjectivity"],
           bins = bins_v,
           labels = labels_v)
    
    discrete_pol_subj_wr_shares["avg_positive_polarity"] = pd.cut(pol_subj_wr_shares["avg_positive_polarity"],
           bins = bins_v,
           labels = labels_v)
    
    discrete_pol_subj_wr_shares["avg_negative_polarity"] = pd.cut(pol_subj_wr_shares["avg_negative_polarity"],
           bins = bins_v,
           labels = labels_v)
    
    discrete_pol_subj_wr_shares["avg_negative_polarity"] = pd.cut(pol_subj_wr_shares["avg_negative_polarity"],
           bins = bins_v_neg,
           labels = labels_v_neg)
    
    discrete_pol_subj_wr_shares["shares"] = v_news["shares"]
    
    discrete_pol_subj_wr_shares.reset_index(drop=True, inplace = True)
    
    discrete_table_pol_positive_subj_wr_shares = pd.pivot_table(discrete_pol_subj_wr_shares,
                                                       index="global_subjectivity",
                                                       columns="avg_positive_polarity",
                                                       values="shares",
                                                       aggfunc="sum").fillna(0)
    
    discrete_table_pol_negative_subj_wr_shares = pd.pivot_table(discrete_pol_subj_wr_shares,
                                                       index="global_subjectivity",
                                                       columns="avg_negative_polarity",
                                                       values="shares",
                                                       aggfunc="sum").fillna(0)
    
    f, axs = plt.subplots(1, 2, figsize = (20, 8))
    
    sns.heatmap(discrete_table_pol_positive_subj_wr_shares, ax = axs[0], cmap="Greens")
    sns.heatmap(discrete_table_pol_negative_subj_wr_shares, ax = axs[1], cmap="Reds")
    
    axs[0].set_title("Average Positivity", size = 18)
    axs[1].set_title("Average Negativity", size=18)
    plt.suptitle("Subjectivity, polarity and their effects on shares", size = 25)
    st.pyplot()
    
    fig = px.histogram(v_news, x="global_sentiment_polarity", y="shares", color="Class_shares2", color_discrete_sequence=Magma7)
    st.plotly_chart(fig)
    
    n_shares_wr_author_chanel = pd.pivot_table(v_news, 
                                           index="Authors", 
                                           columns="Chanel", 
                                           values="shares", 
                                           aggfunc="sum").fillna(0)
    n_shares_wr_author_chanel["cumsum"] = np.sum(n_shares_wr_author_chanel, axis=1)
    n_shares_wr_author_chanel.sort_values(by="cumsum", ascending=False, inplace=True)
    n_shares_wr_author_chanel.drop(labels="cumsum", axis=1, inplace=True)
    top10_n_shares_wr_author_chanel = n_shares_wr_author_chanel.head(10)
    
    fig = px.bar(top10_n_shares_wr_author_chanel, height=400,
                 color_discrete_sequence=Magma7, 
                 title="Number of shares per author and types of articles written")
    st.plotly_chart(fig)
    
    week_ref = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dor = v_news[["Weekday", "shares"]].groupby(by="Weekday").sum().reindex(week_ref)
    means_5000 = (v_news[["Weekday", "shares"]].groupby(by="Weekday").mean().values * 5000)
    
    angles = np.linspace(0.05, 2 * np.pi - 0.05, dor.shape[0], endpoint=False)
    
    GREY12 = "#1f1f1f"
    plt.rcParams.update({"font.family": "Bell MT"})
    
    plt.rc("axes", unicode_minus=False)
    f, ax = plt.subplots(figsize=(7, 10), subplot_kw={"projection": "polar"})
    f.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_theta_offset(1.2 * np.pi / 2)
    ax.set_ylim(-10000000, 28000000)
    
    ax.bar(angles,
           dor["shares"],
           width=0.80,
           color = Pastel2_7,
           alpha = 0.9,
           zorder=10)
    
    ax.vlines(angles, 
              0, 
              25000000, 
              color=GREY12, 
              ls=(0, (4, 4)), 
              zorder=11)
    
    ax.scatter(angles, 
               means_5000, 
               s=60, 
               color=GREY12, 
               zorder=11)
    
    ax.xaxis.grid(False)
    
    
    ax.spines["start"].set_color("none")
    ax.spines["polar"].set_color("none")
    
    ax.set_xticks(angles)
    ax.set_yticklabels([])
    ax.set_yticks([0, 5000000, 10000000, 15000000, 20000000, 25000000])
    
    XTICKS = ax.xaxis.get_major_ticks()
    for tick in XTICKS:
        tick.set_pad(10)
    
    PAD = 10
    ax.text(-0.3*np.pi / 2, 5000000 + PAD, "5M", ha="center", size=8)
    ax.text(-0.3*np.pi / 2, 10000000 + PAD, "10M", ha="center", size=8)
    ax.text(-0.3*np.pi / 2, 15000000 + PAD, "15M", ha="center", size=8)
    ax.text(-0.3*np.pi / 2, 20000000 + PAD, "20M", ha="center", size=8)
    
    ax.set_xticklabels(week_ref, size=12);
    st.pyplot()
    
    
    img_vid_wr_shares = v_news[["num_hrefs", "num_imgs", "num_videos", "shares"]]
    table1_img_vid_wr_shares = pd.pivot_table(img_vid_wr_shares,
                                          index="num_hrefs", 
                                          columns="num_imgs", 
                                          values="shares", 
                                          aggfunc="sum").fillna(0)
    
    fig1 = px.density_heatmap(img_vid_wr_shares, 
                   x="num_hrefs", 
                   y="num_imgs", 
                   z="shares", 
                   marginal_x="histogram", 
                   marginal_y="histogram",
                   histfunc="sum",
                   color_continuous_scale=Purples9)

    fig2 = px.density_heatmap(img_vid_wr_shares, 
                       x="num_hrefs", 
                       y="num_videos", 
                       z="shares", 
                       marginal_x="histogram", 
                       marginal_y="histogram",
                       histfunc="sum",
                       color_continuous_scale=Greens9)
    
    fig3 = px.density_heatmap(img_vid_wr_shares, 
                       x="num_imgs", 
                       y="num_videos", 
                       z="shares", 
                       marginal_x="histogram", 
                       marginal_y="histogram",
                       histfunc="sum",
                       color_continuous_scale=YlOrRd9)
    
    fig1.update_layout(xaxis_range=[0,30])
    fig1.update_layout(yaxis_range=[0,10])
    
    fig2.update_layout(xaxis_range=[0,30])
    fig2.update_layout(yaxis_range=[0,10])
    
    fig3.update_layout(xaxis_range=[0,30])
    fig3.update_layout(yaxis_range=[0,10])
    
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    st.plotly_chart(fig3)

    df = pd.DataFrame()
    df['chanel'] = v_news["Chanel"]
    df['shares'] = v_news['shares']
    df2 = df.groupby('chanel').count().reset_index()
    df =  df.groupby('chanel').sum().reset_index()
    df['count'] = df2['shares']
    df['shares_nb'] = df['shares']/df['count']
    df = df.drop(labels=3, axis=0)
    
    fig = px.bar(df, x='chanel', y='shares_nb', color = 'chanel') 
    st.plotly_chart(fig)
    
    
    df = pd.DataFrame()
    df['shares'] = v_news['shares'] 
    df['lentitle'] = v_news['n_tokens_title']
    df2 = df.groupby('lentitle').count().reset_index()
    df =  df.groupby('lentitle').sum().reset_index()
    df['count'] = df2['shares']
    df['shares_nb'] = df['shares']/df['count']
    
    fig = px.bar(df, x='lentitle', y='shares_nb')
    st.plotly_chart(fig)
    fig = px.scatter(v_news, x='n_tokens_content', y='shares',color = 'Chanel',trendline='ols')
    st.plotly_chart(fig)
    df = pd.DataFrame()
    df['shares'] = v_news['shares'] 
    df['lencorpus'] = v_news['n_tokens_content']
    df2 = df.groupby('lencorpus').count().reset_index()
    df =  df.groupby('lencorpus').sum().reset_index()
    df['count'] = df2['shares']
    df['shares_nb'] = df['shares']/df['count']
    
    fig = px.scatter(df, x='lencorpus', y='shares_nb',trendline='ols')
    st.plotly_chart(fig)
    
def page4():
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")

def page5():
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
    "Overview":main_page ,
    "Data preprocess/discovery": page2,
    "Improve your article": page3,
    "Machine learning/ Deep learning": page4,
    "Predict your success": page5,
    "Libraries": library
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
