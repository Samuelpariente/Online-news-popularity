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
import seaborn as sns


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
import streamlit.components.v1 as components

online = True

if online== True: 
    illustration1 = 'Webapp/illustration1.PNG'
    v_news = 'Webapp/v_news.csv'
    multiTimeline='Webapp/multiTimeline.csv'
    Mlogo = 'Webapp/Mlogo.png'
    OnlineNewsPopularityWithAutorsAndTitles= 'Webapp/OnlineNewsPopularityWithAutorsAndTitles.csv'
    race = "Webapp/Race.mp4"
    ada = "Webapp/graph/graph ada grad class2.png"
    hist = "Webapp/graph/graph Hist grad class2.png"
    adareg = "Webapp/graph/learning curv ada.png"
    knn = "Webapp/graph/Knn.png"
    deep1 = "Webapp/graph/deep1.png"
    deep2 = "Webapp/graph/deep2.png"
    deep3 = "Webapp/graph/deep3.png"
    deep4 = "Webapp/graph/deep4.png"
    tnse = "Webapp/graph/TSNE.png"
    cluster = "Webapp/graph/cluster.png"
    result = "Webapp/result.csv"
    best = "Webapp/graph/best.png"

if online  == False: 
    illustration1 = 'illustration1.PNG'
    v_news = 'v_news.csv'
    multiTimeline='multiTimeline.csv'
    Mlogo = 'Mlogo.png'
    OnlineNewsPopularityWithAutorsAndTitles= 'OnlineNewsPopularityWithAutorsAndTitles.csv'
    race = "Race.mp4"
    ada = "graph ada grad class2.png"
    hist = "graph Hist grad class2.png"
    adareg = "learning curv ada.png"
    knn = "Knn.png"
    deep1 = "deep1.png"
    deep2 = "deep2.png"
    deep3 = "deep3.png"
    deep4 = "deep4.png"
    tnse = "TSNE.png"
    cluster = "cluster.png"
    result = "result.csv"
    best = "best.png"

res = pd.read_csv(result)
magmaS = ['#000000','#2B115E', '#3B0F6F', '#8C2980', '#DD4968', '#FD9F6C', '#FBFCBF']
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
    st.markdown("# To go further  ")
    """
    ### Improve the Dataset 
    
    While creating a predictive model for the number of shares per article (discrete or continuous), we have been processing the data to get the most correct prediction possible. We went through outlier management, intelligent selection of study variables, creation of a new variable, as well as vectorization of one of the qualitative classes.  Then we studied several models that give us an accuracy of 67.3% for the binary prediction, 40.3% for the 4 class prediction and 11% for the regression.  
    We can therefore question the quality of the data. First, a variable that could explain these results is the health of the Mashable website. The number of shares is very correlated with this information and even if we managed to retrieve some information from google trend, it is not precise enough to take it into account. 

    ### Thank you
    
    We would like to thank our Python for Data Analysis teacher SABRY Abdellah for his excellent advice throughout the project
    
    
    ### libraries :

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
    st.markdown("## Datasets ")
    """
    In the context of our project we studied the online-news-popularity dataset of mashable articles. Our project contains 2 csv datasets loaded in this section :

    - **Online News Popularity Dataset**
      - 63 columns | 39644 rows
      - Base dataset 
    """
    news = pd.read_csv(OnlineNewsPopularityWithAutorsAndTitles)
    news.iloc[0:5]
    """
    
    - **Timeline Dataset**
      - 2 columns | 103 rows
      - Contains "Mashable.com" internet user's flow for 103 dates 
    
    """

    timeline = pd.read_csv(multiTimeline)
    timeline.iloc[0:5]
    
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

    """
    st.image(best)
    """
    6. **3 predictable Variables**
    """
    
    fig = px.histogram(v_news, x="shares",nbins = 50, color_discrete_sequence = [Magma7[2]], 
             title = "Continuous shares distribution")
    st.plotly_chart(fig)

    fig =px.histogram(v_news, x="Class_shares1", color_discrete_sequence = [Magma7[2]], 
             title = "Binary discretized shares distribution")
    st.plotly_chart(fig)
    

    fig = px.histogram(v_news, x="Class_shares2", color_discrete_sequence = [Magma7[2]], 
             title = "Quantile discretized shares distribution")
    st.plotly_chart(fig)
      
    
def page2():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown("# Data discovery ")
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
              color = lda["Chanel"].head(200), opacity=0.7, color_discrete_sequence=magmaS)
        

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
        fig = px.histogram(v_news, x="global_sentiment_polarity", y="shares", color="Class_shares2", color_discrete_sequence=magmaS[:4], title =  "Global sentiment polarity with respect to the number of shares")
        st.plotly_chart(fig)
    else:
        fig = px.histogram(v_news, x="global_sentiment_polarity", y="shares", color="Class_shares1", color_discrete_sequence=magmaS[:2], title = "Global sentiment polarity with respect to the number of shares")
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
                 color_discrete_sequence=magmaS, 
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
           color = magmaS,
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
    
    fig = px.bar(df, x='chanel', y='shares_nb', color = 'chanel', color_discrete_sequence=magmaS)
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
    
    fig = px.scatter(v_news, x='n_tokens_content', y='shares',color = 'Chanel',trendline='ols', color_discrete_sequence=magmaS, title="Scatter plot of shares with respect to the chanel and the amount of words in the article's body")
    st.plotly_chart(fig)
    
def page4():
    st.markdown("## 4 - Prediction")
    s = """ The following section aims at predicting the success of an article in the most precise way possible. To do so, we will use multiple types of prediction algorithms. We will use 2 class of algorithms, **Machine Learning** and **Deep Learning** ones. Amongst them, We find **classification** and **regression** algorithms that we will both use as well. In addition, we will use the Auto-Sklearn library to maximize our result.    

⚙ Auto-sklearn is a program that tries multiple algorithms from sk-learn library. You input the amount of time that you allow the autoMl to run for, you also input the type of algorithm that you desire (Regression or Classificaition). In the end, the program will output a list of its best found Machine Learning models with the best found hyperparameters for each one.

⚡Each Algorithm's functionning is detailed in its own subpart. 

***
***

I. Machine Learning
1. Regression
- First, we compute an AutoML computation with Auto-Sklearn library. 
- The AutoML indicates that <ins>ARDRegression</ins> is the most fitted model. Consequently, we compute an ARDRegression with a grid Search of our own.
- 🥇 : 11.1%

***
2. Binary Classification (Low or High)
- Similarly to the regression part, we compute an AutoML.
- <ins>Gradient Boosting</ins>, <ins>AdaBoost</ins> and <ins>Random Forest</ins> stand out as the **best Binary Classification algorihtms**. So we compute them independently from the AutoML with a targeted grid search. 
- In addition, we deploy a <ins>KNN</ins> algorithm to evalutate its performances.
- 🥇 : 67.3%

***

3. Quantile Classification (Very Low, Low, High, Very High)
- This subsection is very similary to the binary classification in terms of the method we use. 
- After computing the AutoML, we find out that <ins>HistogramGradientBoosting</ins> and <ins>Random Forest</ins> are the two most appropriate algorithms for this classification.
- 🥇 : 40.2%

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
- 🥇 : <ins>Best Accuracy</ins> on the test set was **36.1%**.
        
    """
    st.markdown(s, unsafe_allow_html=True)
    
    """
    ## I. Machine Learning
    ### 1. Regression
    - 🥇 :11.1%

    ##### ARDRegression (Automatic Relevance Determination Regression)
    🧠 ARD Regression is a Machine Learning model derived from [Bayesian Ridge Regression](https://scikit-learn.org/0.17/modules/linear_model.html#bayesian-ridge-regression). 
    """

    st.image(adareg) 
    st.metric(label="Accuracy", value="11.1%")
    """
    ### 2. Binary Classification (Low or High)
    - 🥇 :67.3%
        
    
    ##### AUTOML
    """
    body = """
    automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=30000,per_run_time_limit=3600, memory_limit = 25000)
    automl.fit(x_train_scale, y_train)

    """
    st.code(body, language="python")
    
    s = """
    ##### Histogram Gradiant boosting
     🧠 **Binning** is the segmentation and convertion to numerical value of a set of data, that is what we do when we create a histogram. This same binning concept is <ins>applied to the Decision Tree</ins> (DT) algorithm. By reducing the number of features, it will be used to increase the algorithm’s speed. As a result, the same notion is employed in DT by grouping with histograms, which is known as the HGB classifier.

    📚 Source : [AnalyticsVidhya](https://www.analyticsvidhya.com/blog/2022/01/histogram-boosting-gradient-classifier/)
    
     """
    st.markdown(s, unsafe_allow_html=True)
    body= """
    model = HistGradientBoostingClassifier(early_stopping=True, l2_regularization=1e-10,
                                 learning_rate=0.13046552565826827, max_iter=64,
                                 max_leaf_nodes=41, min_samples_leaf=40,
                                 n_iter_no_change=6, random_state=1,
                                 warm_start=True)
    """
    st.code(body, language="python")
    st.metric(label="Accuracy", value="67.3%")
    st.image(hist) 
    
    s = """
    ##### AdaBoost
    🧠 AdaBoost, short for Adaptive Boosting, is a statistical classification meta-algorithm formulated by Yoav Freund and Robert Schapire in 1995, who won the 2003 Gödel Prize for their work. Usually, AdaBoost is presented for **binary classification**, although it can be generalized to multiple classes or bounded intervals on the real line.

    AdaBoost is adaptive in the sense that **subsequent weak learners are tweaked in favor of those instances misclassified by previous classifiers**. In some problems it can be less <ins>susceptible to the overfitting</ins> problem than other learning algorithms. The individual learners can be weak, but as long as the performance of each one is slightly better than random guessing, the final model can be proven to converge to a strong learner.

    📚 Source : [AdaBoost Wikipedia](https://en.wikipedia.org/wiki/AdaBoost)
    """
    st.markdown(s, unsafe_allow_html=True)
    body= """
   model = AdaBoostClassifier(algorithm='SAMME',
                   base_estimator=DecisionTreeClassifier(max_depth=2),
                   learning_rate=0.03734246906377268, n_estimators=416,
                   random_state=1)
    """
    st.code(body, language="python")
    st.metric(label="Accuracy", value="66.0%")
    st.image(hist) 
    
    s = """
    ##### LightGBM  
    🧠 LightGBM, short for light gradient-boosting machine, is a free and open-source distributed gradient-boosting framework for machine learning, originally developed by Microsoft. It is based on decision tree algorithms and used for ranking, classification and other machine learning tasks. The development focus is on performance and scalability.

    📚 Source : [Wikipedia LightGBM](https://en.wikipedia.org/wiki/LightGBM)
    """
    st.markdown(s, unsafe_allow_html=True)
    body= """
    lgbm=lgb.LGBMClassifier(n_estimators= 100, boosting_type= 'gbdt', colsample_bytree= 0.8, learning_rate= 0.09, max_depth=30)
    """
    st.code(body, language="python")
    st.metric(label="Accuracy", value="67.2%")
    
    s = """
    ##### XGboost
    
    XgBoost stands for Extreme Gradient Boosting, which was proposed by the researchers at the University of Washington. It is a library written in C++ which optimizes the training for Gradient Boosting.
    🧠 In this algorithm, decision trees are created in sequential form. Weights play an important role in XGBoost. Weights are assigned to all the independent variables which are then fed into the decision tree which predicts results. The weight of variables predicted wrong by the tree is increased and these variables are then fed to the second decision tree. These individual classifiers/predictors then ensemble to give a strong and more precise model. It can work on regression, classification, ranking, and user-defined prediction problems.

    📚 Source : [GeeksForGeeks](https://www.geeksforgeeks.org/xgboost/)
    """
    st.markdown(s, unsafe_allow_html=True)
    body= """
    model = XGBClassifier(
    objective= 'binary:logistic',
    booster = 'gbtree',
    eval_metric = 'auc',
    eta = 0.08,
    subsample = 0.9,
    colsample_bytree=0.9,
    max_depth = 20,
    base_score = y_train.mean(),
    seed = 666,
    early_stopping_rouds = 50
    )
    """
    st.code(body, language="python")
    st.metric(label="Accuracy", value="66.8%")
    """
    ##### Random Forest
    🧠 Random Forest is a bagging algorithm that has the specificity to **take a subset of the explainatory variables** for each tree that it builds unlike classical bagging that takes all of them. This **prevents the different learning trees from being correlated**.

    📚 Source : [Random Forest Wikipedia](https://fr.wikipedia.org/wiki/For%C3%AAt_d%27arbres_d%C3%A9cisionnels)
    """
    body= """
    model = RandomForestClassifier(max_features=7, n_estimators=512, n_jobs=1,
                         random_state=1, warm_start=True)
    """
    st.code(body, language="python")
    st.metric(label="Accuracy", value="66.6%")
    
    """
    ##### KNN
    🧠 KNN as a classification algorithm allows determines the class of an unknown individual based on the class of its K nearest neighbors.
    """
    body= """
    k_range = np.arange(1,100)
    accuracy = []
    
    for n in k_range:    
        neigh = KNeighborsClassifier(n_neighbors=n, n_jobs=-1)
    
        neigh.fit(x_train_scale, y_train)  
    
        # predict the result
        y_pred = neigh.predict(x_test_scale)
        accuracy.append(100*accuracy_score(y_pred, y_test))
    """
    st.code(body, language="python")
    st.metric(label="Accuracy", value="63.3%")
    st.image(knn) 
    """
    ##### Result
    """
    fig = go.Figure(data=[
    go.Bar(name='Accuracy', x=res['Name'], y=res['Accuracy']),
    go.Bar(name='Precision', x=res['Name'], y=res['Precision']),
    go.Bar(name='Recall', x=res['Name'], y=res['Recall']),
    go.Bar(name='F1', x=res['Name'], y=res['F1']),
    ])
    # Change the bar mode
    fig.update_layout(barmode='group')
    st.plotly_chart(fig)
    """
    ## 3 - Quantile Classification (Very low, low, high, very high)
    🥇 :40.2%
    
    ##### Hist Gradient Boosting
    
    🧠 If you want informations about the functionning of this algorithm please check the Binary Classification section in which it is explained.
    """    
    body= """
    model = HistGradientBoostingClassifier(early_stopping=True,
                                 l2_regularization=2.506856350040198e-06,
                                 learning_rate=0.04634380160611007, max_iter=512,
                                 max_leaf_nodes=11, min_samples_leaf=41,
                                 n_iter_no_change=17, random_state=1,
                                 validation_fraction=None, warm_start=True)
    """
    st.code(body, language="python")
    st.metric(label="Accuracy", value="40.0%")
    """
    ##### Random Forest
    
    🧠 If you want informations about the functionning of this algorithm please check the Binary Classification section in which it is explained.
    """
    body= """
    model = RandomForestClassifier(criterion='entropy', max_features=11,
                         min_samples_leaf=17, min_samples_split=5,
                         n_estimators=512, n_jobs=1, random_state=1,
                         warm_start=True)
    """
    st.code(body, language="python")
    st.metric(label="Accuracy", value="40.2%")
    """
    ## 4. Clustering
    ##### PCA and TSNE Transformation
    """
    body= """
    tsne = TSNE(n_components=2, n_iter=300)
    reduced_tsne = tsne.fit_transform(x_train_scale)
    
    # plotting the clusters TSNE
    plt.figure(figsize=(10,10))
    plt.plot(reduced_tsne[:,0], reduced_tsne[:,1], 'r.')
    plt.title('TSNE Transformation')
    plt.show()
    """
    st.code(body, language="python")
    st.image(tnse) 
    """
    ##### KMeans
    
    🧠 k-means clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster.


    📚 Source : [K-means Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering)
    """
    body= """
    kmeans=KMeans(init='k-means++',n_clusters=5)
    kmeans.fit(reduced_tsne)
    kmeans_preds=kmeans.predict(reduced_tsne)

    """
    st.code(body, language="python")
    st.image(cluster)
    
    """
    ## 5. Deep Learning
    
    ⚙ How do they work ?
    - The Sequential API allows to create models layer-by-layer for most problems.
    - We can set the number of of neurones for each layers, define their activation functions, their input or output format.

    💭 Why do we use it ?
    - The sequential API allows to create Deep Learning models in a very intuitive manner trough array like inputations.
    - In addition, the sequential fashion permits more flexibility.
    
    ![image.png](data:image/png;base64,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)
    ##### Regression Neural Network
    
    First Model :
    """
    body= """
    model = Sequential()
    model.add(Dense(24, activation='relu', input_shape=(63,)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse',
        metrics=['mae'],
        optimizer = optimizers.RMSprop(0.01))

    """
    st.code(body, language="python")
    st.image(deep1)
    
    """
    Second Model :
    """
    body= """
    model2 = Sequential()
    model2.add(Dense(5, activation='relu', input_shape=(63,)))
    model2.add(Dense(25, activation='relu'))
    model2.add(Dropout(0.1))
    model2.add(Dense(25, activation='relu'))
    model2.add(Dense(1))
    model2.compile(loss='mse',
              metrics=['mae'],
              optimizer = optimizers.RMSprop(0.003))

    """
    st.code(body, language="python")
    st.image(deep2)
    
    """
    ##### Binary Classification Neural Network
    
    High/low
    """
    body= """
    model_reg1 = Sequential()
    model_reg1.add(Dense(500, activation='relu', input_shape=(63,)))
    model_reg1.add(Dense(100, activation='relu'))
    model_reg1.add(Dense(50, activation='relu'))
    model_reg1.add(Dense(2, activation='sigmoid'))
    
    # Compile the model
    model_reg1.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    """
    st.code(body, language="python")
    st.metric(label="Accuracy", value="58.4%")
    st.image(deep3)
    """
    Very low/ low/ high/ very high
    """
    body= """
    model_reg2 = Sequential()
    model_reg2.add(Dense(8, input_shape=(63,), activation='relu'))
    model_reg2.add(Dense(4, activation='softmax'))
    
    # Compile the model
    model_reg2.compile(loss='categorical_crossentropy', 
                   metrics=['accuracy'], 
                   optimizer=optimizers.Adam(0.004))
    """
    st.code(body, language="python")
    st.metric(label="Accuracy", value="28.3%")
    st.image(deep4)
    
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
    "To go further  ": library
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
