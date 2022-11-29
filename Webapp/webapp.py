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

from bokeh.palettes import Spectral5
from bokeh.palettes import PRGn7
from bokeh.palettes import Magma7
from bokeh.palettes import Magma10
from bokeh.palettes import Magma6
from bokeh.palettes import Viridis
from bokeh.palettes import Pastel2_7
from bokeh.palettes import Purples9
from bokeh.palettes import Greens9
from bokeh.palettes import YlOrRd9
from bokeh.transform import factor_cmap, transform
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show

from collections import OrderedDict

online = True

if online== True: 
    illustration1 = 'Webapp/illustration1.PNG'
    v_news = 'Webapp/v_news.csv'
    multiTimeline='Webapp/multiTimeline.csv'
    Mlogo = 'Webapp/Mlogo.png'
    OnlineNewsPopularityWithAutorsAndTitles= 'Webapp/OnlineNewsPopularityWithAutorsAndTitles.csv'
    race = "Webapp/Race.mp4"
    
if online  == False: 
    illustration1 = 'illustration1.PNG'
    v_news = 'v_news.csv'
    multiTimeline='multiTimeline.csv'
    Mlogo = 'Mlogo.png'
    OnlineNewsPopularityWithAutorsAndTitles= 'OnlineNewsPopularityWithAutorsAndTitles.csv'
    race = "Race.mp4"

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
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown("# Data discovery ")
    st.sidebar.markdown("Data discovery ")
    """
    This section is dedicated to understanding how our data behaves through mutliple graphs. Our work in this part can be split in two categories :

    - Univariate Visualization :
      - Allows use to determine the behavior of our variables.
      - Can act as a reference in the future and more in depth studies of this notebook.
    
    - Multivariate Visualization :
      - Permits to comprehend to what extent variables interact.
      - Is the first to understanding the data troughoutly and finding the problematic of this project.
      

      """
    st.markdown("##### Repartition of releases day through the week")
    st.markdown("By analysing this plot, we notice that the **3 most common day of release** are <ins>Tuesday</ins>, <ins>Wednesday</ins> and <ins>Thursday</ins> while the 2 less common are <ins>Saturday</ins> and <ins>Sunday</ins>.", unsafe_allow_html=True)
    week_ref = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    Weekday_counts = v_news.groupby(by="Weekday").Weekday.count().reindex(week_ref)
    index = list(Weekday_counts.index)
    values = list(Weekday_counts.values)
    
    source = ColumnDataSource(data=OrderedDict(weekday=index, counts=values))
    
    p = figure(x_range=index, y_range=(0, np.max(values)+1000), height=350, title="Release Day Frequencies",
               toolbar_location=None, tools="hover", tooltips="@weekday: @counts")
    
    p.vbar(x='weekday', top='counts', width=0.9, source=source, 
           line_color='white', fill_color=factor_cmap('weekday', palette=Magma7, factors=index))
    
    p.xgrid.grid_line_color = None
    st.bokeh_chart(p)

    st.markdown("##### Repartition of Main topics amongst Mashable articles.")
    st.markdown("The spiderplot bellow shows us the <ins>Business</ins>, <ins>World</ins>, <ins>Tech</ins> and <ins>Entertainement</ins> are the **4 most common topics** on Mashable.", unsafe_allow_html=True)
    chanel = v_news.groupby(by="Chanel").Chanel.count()

    fig = go.Figure(data=go.Scatterpolar(
      r=chanel.values,
      theta=chanel.index,
      fill='toself',
      name='Frequencies of Chanels',
      marker_color=Magma7[2], 
      line_color = Magma7[2],
      marker_line_color="black",
    ))
    
    fig.update_layout(
      title={
          'text': "Frequencies of Chanels", 
          'y': 0.9,
          'x': 0.5
      },
      polar=dict(
        radialaxis=dict(
          visible=True, 
        ),
      ),
      showlegend=False, 
      
    )


    st.plotly_chart(fig)
    
  
    st.markdown("##### Most prolific authors")
    st.markdown("The pie plow bellow shows the 10 most prolific Authors on Mashable.", unsafe_allow_html=True)
    
    autnb = v_news.groupby('Authors').count()['Titles'].sort_values(ascending = False)[:10]
    autnb =autnb.reset_index()
    pie_values=autnb["Titles"]
    pie_labels=autnb["Authors"]
    explode = [0.1 if i == 0 else 0 for i in range(10)]
    
    fig1, ax1 = plt.subplots()
    
    ax1.pie(pie_values,  labels=pie_labels,
            shadow=True, startangle=0, colors=Magma10, explode=explode)
    plt.title("10 Most prolific Authors on Mashable")
    st.pyplot()
    
    
    
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
    st.markdown("#### Most Frequent words ")
    """
    ##### Most common topics
    In order to visualize the most common topics of articles, we studied the frequency of words in titles. We present it here as a word cloud. 

    ❗ Notice that the 10 most frequent word are colored in red. Amongst them, we find **3 of the 5 GAFAM** and a famous social media, **twitter** ❗ 
    """
    st.pyplot()
    """
    Plotting all of our 62 variables wouldn't have been much insightfull. In this context, we chose to plot the totality of our quantitatives variables. We used these graphs as a reference during our work, in the event where we require the distribution of a specific variable.
    """
    v_news.hist(figsize=(20,20), color=Magma7[2])
    st.pyplot()
    
    
    """
    This subsection aims at visualizing the proximity of articles depending on chanel to which they are related. Luckly, the dataset contains LDA (Latent Dirichlet Allocations) topics that can be used to evaluate the relative positions of 2 to n articles.

    However, we have 4 LDA columns, meaning 4 LDA topics but we can only visualize data in 3 dimensions. Given that only the relative position of the points (articles) in the graph matters, we can compute a PCA to obtain a 3 dimensional graphic. 
    
    When selecting 3 components for the PCA, 84% of the data is explained which is higly satisfying..
    
    By looking at the graph, we can notice that the Chanel of an article (its general category) is certainly correlated to its Latent Dirichelet topics proximity. For instance, red points (business related articles) are somewhat from one another. The same goes for the other Chanels.
    """
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
    
    # tight layout
    fig = px.scatter_3d(x=pca_values[:200, 0], y=pca_values[:200, 1], z=pca_values[:200, 2],
              color = lda["Chanel"].head(200), opacity=0.7, color_discrete_sequence=Magma7)
        

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        legend_title="Chanels", 
        title={
        'text': "3D plot of articles in PCA reduced LDA topics space", 
          'y': 0.9,
          'x': 0.45
        }
    )
    
    st.markdown("## Proximity of chanels")
    st.plotly_chart(fig)
    

    """
    This section Echos to the TOP 10 most prolific authors seen in the univariate section of the visualization. In fact, here we show the evolution over time of the authors having the most shares. As you can see the race is pretty stacked 🛫
    """
    
    
    
    video = open(race, 'rb').read()
    st.video(video)
    
    """
    This correlation heatmap shows use the correlation of each variable with respect to all the others. As we can see some variables are highlt correlated between each others. However, most variables are uncorrelated.

    ❗Note that the correlated variables won't be taken into account for the Machine and Deep learning part of our study.
    """
    corr = v_news.corr()
    fig = px.imshow(corr,color_continuous_scale='RdBu_r', text_auto=True)
    st.plotly_chart(fig)
    
def page3():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    """
    ## 3 - How to improve your articles ?
    
    ##### Average polarity and subjectivity, how they interact and affect shares of a given article
    
    The 3 graphics that follows, 2 heatmap and 1 histogram, consist of insights about the link between the emotion put in an article, its subjectivity and how its affects the shares of the article.

    ***
    We notice that the the most shared articles has a medium subjectivity (around 0.4 to 0.5) and are either slightly positive of negative regarding their content. However, based on the third graph, we can infer that **positive articles are most likely to be shared a lot**.
    """
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
    
    
    genre = st.radio(
    "Numbers of classes: ",
    ('2 classes', '4 classes'))

    if genre == '4 classes':    
        fig = px.histogram(v_news, x="global_sentiment_polarity", y="shares", color="Class_shares2", color_discrete_sequence=Magma7, title = "Global sentiment polarity with respect to the number of shares")
        st.plotly_chart(fig)
    else:
        fig = px.histogram(v_news, x="global_sentiment_polarity", y="shares", color="Class_shares1", color_discrete_sequence=Magma7, title = "Global sentiment polarity with respect to the number of shares")
        st.plotly_chart(fig)
    """
    From these graphs, we conclude that articles with more positive polarity tend to be more appreciated. (biased interpretation)
    """
    st.markdown("##### Number of shares per author and type of article")
    st.markdown("The graph that you will find bellow aims at looking at the repartition of chanels' articles of the most shared authors so as to understand what subjects are more likeky to get you great amounts of shares. it appears that the <ins>key of famous authors</ins> is to create articles in multiple chanels to **diversify their content**.", unsafe_allow_html=True)
    
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
    
    """
    ##### Shares with respect to day of release
    The following graph shows the amount of shares with respect to the day of the week.
    Source for the graph : https://www.python-graph-gallery.com/web-circular-barplot-with-matplotlib
    """
    week_ref = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dor = v_news[["Weekday", "shares"]].groupby(by="Weekday").sum().reindex(week_ref)
    means_5000 = (v_news[["Weekday", "shares"]].groupby(by="Weekday").mean().values * 2500)
    
    angles = np.linspace(0.05, 2 * np.pi - 0.05, dor.shape[0], endpoint=False)
    
    GREY12 = "#808080"
    plt.rcParams.update({"font.family": "Bell MT"})
    
    plt.rc("axes", unicode_minus=False)
    f, ax = plt.subplots(figsize=(7, 10), subplot_kw={"projection": "polar"})
    f.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_theta_offset(1.2 * np.pi / 2)
    ax.set_ylim(-5000000, 12000000)
    
    ax.bar(angles,
           dor["shares"],
           width=0.80,
           color = Magma7,
           alpha = 0.9,
           zorder=10)
    
    ax.vlines(angles, 
              0, 
              12000000, 
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
    ax.set_yticks([0, 4000000, 8000000, 12000000])
    
    XTICKS = ax.xaxis.get_major_ticks()
    for tick in XTICKS:
        tick.set_pad(10)
    
    PAD = 10
    ax.text(-0.3*np.pi / 2, 4000000 + PAD, "5M", ha="center", size=12)
    ax.text(-0.3*np.pi / 2, 8000000 + PAD, "10M", ha="center", size=12)
    ax.text(-0.3*np.pi / 2, 12000000 + PAD, "15M", ha="center", size=12)
    
    ax.set_xticklabels(week_ref, size=12);
    _=plt.title("BarPlot of shares with respect to the day of the week")
    st.pyplot()
    
    """
    ##### How additionnal medias (links, images and videos) affect shares
    The 3 following heatmap tend to indicate what are the optimal numbers of medias to include in your article. 

    From this study, the optimal parameters seem to be :
    - Between 3 and 5 hyperlinks.
    - No images.
    - No videos.
    
    ❗this result may be biased because there are less articles with videos and images on mashable.com
    """
    
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
    """
    ##### Number of shares with respect of topic
    
    """
    st.markdown("Interpreting the barplot that follows, we notice that all chanels are relatively close in terms of numbers of shares. However, <ins>Lifestyle</ins> and <ins>Social Media</ins> chanels seem to be sightly dominant.",unsafe_allow_html=True)
    df = pd.DataFrame()
    df['chanel'] = v_news["Chanel"]
    df['shares'] = v_news['shares']
    df2 = df.groupby('chanel').count().reset_index()
    df =  df.groupby('chanel').sum().reset_index()
    df['count'] = df2['shares']
    df['shares_nb'] = df['shares']/df['count']
    df = df.drop(labels=3, axis=0)
    
    fig = px.bar(df, x='chanel', y='shares_nb', color = 'chanel', color_discrete_sequence=Magma6)
    st.plotly_chart(fig)
    
    """
    ##### Length of Titles with respect to shares
    
    Given the study on the title's length of articles we infer that the ideal title should :
    - Be 4 to 17 words long (even if the difference isn't immense when titles aren't in this range).
    - Contain words between 6 to 15 characters. However, the complexity of the word doesn't seem to affect the number of shares in a tremendous manner so **come as you are** as far as the title's vocabulary is concerned.
    
    ❗In this part, we suppose that the length of a word is correlated to its complexity. Consequently, we may analyze the result of the subsection through the complexity aspect.
    """
    
    df = pd.DataFrame()
    df['shares'] = v_news['shares'] 
    df['lentitle'] = v_news['n_tokens_title']
    df2 = df.groupby('lentitle').count().reset_index()
    df =  df.groupby('lentitle').sum().reset_index()
    df['count'] = df2['shares']
    df['shares_nb'] = df['shares']/df['count']
    
    fig = px.bar(df, x='lentitle', y='shares_nb', title="Shares with respect to the Bar plot of the length of Titles")
    fig.update_traces(marker_color=Magma7[2])
    st.plotly_chart(fig)
    
    """
    ##### Length of bodies with respect to shares
    """
    st.markdown("In the section, we find out that the optimal length is between <ins>1000 and 3000 words long</ins>. Most articles fall in this category and the amount of shares seems to follow through !",unsafe_allow_html=True)
    
    
    df = pd.DataFrame()
    df['shares'] = v_news['shares'] 
    df['lencorpus'] = v_news['n_tokens_content']
    df2 = df.groupby('lencorpus').count().reset_index()
    df =  df.groupby('lencorpus').sum().reset_index()
    df['count'] = df2['shares']
    df['shares_nb'] = df['shares']/df['count']
    
    fig = px.scatter(df, x='lencorpus', y='shares_nb',trendline='ols')
    fig.update_traces(marker_color=Magma7[2])
    st.plotly_chart(fig)
    
    fig = px.scatter(v_news, x='n_tokens_content', y='shares',color = 'Chanel',trendline='ols', color_discrete_sequence=Magma7, title="Scatter plot of shares with respect to the chanel and the amount of words in the article's body")
    st.plotly_chart(fig)
    
def page4():
    st.markdown("## 4 - Prediction")
    s = """ The following section aims at predicting the success of an article in the most precise way possible. To do so, we will use multiple types of prediction algorithms. We will use 2 class of algorithms, **Machine Learning** and **Deep Learnin** ones. Amongst them, We find **classification** and **regression** algorithms that we will both use as well. In addition, we will use the Auto-Sklearn library to maximize our result.    

    ⚙ Auto-sklearn is a program that tries multiple algorithms from sk-learn library. You input the amount of time that you allow the autoMl to run for, you also input the type of algorithm that you desire (Regression or Classificaition). In the end, the program will output a list of its best found Machine Learning models with the best found hyperparameters for each one.
    
    ⚡Each Algorithm's functionning is detailed in its own subpart. 
    
    ***
    ***
    
    I. Machine Learning
    
    1. Regression
    - First, we compute an AutoML computation with Auto-Sklearn library. 
    - The AutoML indicates that <ins>ARDRegression</ins> is the most fitted model. Consequently, we compute an ARDRegression with a grid Search of our own.
    - 🥇 : 
    
    ***
    2. Binary Classification (Low or High)
    - Similarly to the regression part, we compute an AutoML.
    - <ins>Gradient Boosting</ins>, <ins>AdaBoost</ins> and <ins>Random Forest</ins> stand out as the **best Binary Classification algorihtms**. So we compute them independently from the AutoML with a targeted grid search. 
    - In addition, we deploy a <ins>KNN</ins> algorithm to evalutate its performances.
    - 🥇 :
    
    ***
    
    3. Quantile Classification (Very Low, Low, High, Very High)
    - This subsection is very similary to the binary classification in terms of the method we use. 
    - After computing the AutoML, we find out that <ins>HistogramGradientBoosting</ins> and <ins>Random Forest</ins> are the two most appropriate algorithms for this classification.
    - 🥇 : 
    
    ***
    
    4. Clustering
    - Unlike the previous algorithms we computed, clustering is part of the  unsupervised learning class of Machine Learning algorithms.  
    - We compute a PCA (Principal Component Analysis) and TNSE (Tense classification) transformation.
    - After that, we compute a K-mean to then visualize our clusterized data.
    
    ***
    ***
    
    II. Deep Learning
    
    1. Regression Neural Network
    - We use a Keras Sequential neural network to compute a regression on our data. To make it work, we mostly use Dense neural layers with Relu activation (Rectified Linear Unit). Only the last layer doesn't have and activation function because it is the output layer.
    - 🥇 : <ins>Best Mean Absolute error</ins> on the test set was **1650**.
    
    ***
    
    2. Binary Classification Neural Network (Low or High)
    - As for Regression we use a Keras Sequential network. However this time, the output layer has to be sigmoid and contains 2 neurons because the input was One hot encoded on 2 columns.
    - 🥇 : <ins>Best Accuracy</ins> on the test set was **61.3%**.
    
    ***
    
    3. Quantile Classification Neural Network (Very Low, Low, High, Very High)
    - Finally the quantile classification differs from the binary one because we use a SotfMax output layer containing 4 neurones.
    - 🥇 : <ins>Best Accuracy</ins> on the test set was **36.1%**
        
    """
    st.markdown(s, unsafe_allow_html=True)
    
    
    
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
    "Data discovery": page2,
    "Improve your article": page3,
    "Machine learning/ Deep learning": page4,
    "Predict your success": page5,
    "Libraries": library
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
