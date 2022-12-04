# Online-news-popularity - Complete study

🔗 [Webapp](https://samuelpariente-online-news-popularity-webappwebapp-s3npdr.streamlit.app/)

### 👨‍🔬 Our study's context and contributors

In the context of our 4th year python class's final project, we decided to study the Online-News Popularity Dataset.
Two person worked on this project <ins>Samuel Pariente</ins> and <ins>Marius Ortega</ins>.

### 📁 Dataset presentation

This dataset was initaly published in 2015 and contains data regarding 39797 articles released on the website [mashable.com](https://mashable.com/) between **2013 and 2015**. It is composed as follows :
- 61 variables in total
- 58 predictive variables
- 2 non predictive variables
- 1 target variable (shares)

Its is worth mentionning that the dataset doesn't have any NA's. However, multiple step of cleaning were still mandatory before starting any predictive process on it.

### 📚 The documents to your disposal
Our work led to the creation of multiple documents, all accessible from this GitHub :
- [A notebook](https://github.com/Samuelpariente/Online-news-popularity/tree/main/Notebook%20version) : You can find in it the totality of our work on the dataset. All of our scientific procedures are detailed there.
- [Improved Dataframes](https://github.com/Samuelpariente/Online-news-popularity/tree/main/Dataframes) : Given that we scrapped additional data from Mashable and internet in general, our base dataset has more columns than the initial one. We can mention <ins>Author's name</ins>, <ins>Title of the article</ins> or the <ins>website trafic</ins>.
- [Powerpoint presentation](https://) : This powerpoint stands as a report to our teachers. We presented it as the final step of the project.
- [Interactive Webapp](https://samuelpariente-online-news-popularity-webappwebapp-s3npdr.streamlit.app/) : Deployed with Streamlit, this webapp is an handy way to introduce people to our work. In contains the same information as our notebook expect the code. In addition, the webapp has a __predict your success__ section that allows you to predict in real time the future of your article's popularity. To do so, it is linked to a API powered on AWS served and containing out most effective machine learning model. 

### 🔍 The structure of the project

The project is separated in 4 main sections : 
- Preprocessing 
- Data Discovery (Univariate and Bivariate analysis)
- Optimize an article's success (Data Insights)
- Prediting the success of an article (Machine Learning and Deep Learning Models)

💡 If you want any additional details regarding our project, feel free to take a look at the documents we mentionned.  
