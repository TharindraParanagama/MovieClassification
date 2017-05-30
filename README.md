# MovieClassification
A web based tool using machine learning to determine the IMDb rating of a given the movie's features.

The approach to this project was as foolows:

1.Choosing a dataset 

2.Data preprocessing(Feature selection,dimesion reduction ec.) 

3.Determining the inputs & outputs of the network

4.Feature clustering to determine the most relevant output labels

5.Creation of a multi layer neural network

6.Training the network

7.Testing the network

8.Predicting the rating for a new data

9.Data visualization

10.Creation of a webpage to display results

Choosing a Dataset

We found a ideal dataset for our purpose on a famous website for dataset which is Kaggle.com.The link to the dataset is as follows:https://www.kaggle.com/deepmatrix/imdb-5000-movie-dataset.

Brief introduction about the dataset

the creator(chuansun76) of this datset had scraped 5000+ movies from IMDB website using a Python library called "scrapy".After the web scapping this datset consisted of 28 variables(or columns) for 5043 movies and 4906 posters (998MB), spanning across 100 years in 66 countries. There are 2399 unique director names, and thousands of actors/actresses. Below are the 28 variables:

"movie_title" "color" "num_critic_for_reviews" "movie_facebook_likes" "duration" "director_name" "director_facebook_likes" "actor_3_name" "actor_3_facebook_likes" "actor_2_name" "actor_2_facebook_likes" "actor_1_name" "actor_1_facebook_likes" "gross" "genres" "num_voted_users" "cast_total_facebook_likes" "facenumber_in_poster" "plot_keywords" "movie_imdb_link" "num_user_for_reviews" "language" "country" "content_rating" "budget" "title_year" "imdb_score" "aspect_ratio"

Data preprocessing

First we chose the most relevant features which we thought would be meaningful features that would directly impact the rating of a movie.

Feature selection/Inputs 

num_critic_for_reviews(as # of critics)
movie_facebook_likes(as Total likes for the movie)
director_facebook_likes(as Director’s FB likes)
actor_1_facebook_likes(as Main Actor’s FB likes)
gross(as Revenue from the movie)

Outputs

We chose three classes which were poor, moderate and excellent.These were denoted by 0,1 and 2 respectively.

We also had to get rid of missing values for which we used a code to check whether any of the cells contained missing values if so we eliminated all such records.This reduced the dataset to 4149 records.

Clustering 

Now we have to determine the labels of the features for that we used K-Means clutering algorithm based on the similarities of the input features.This algorithm was easyly implemented using scikit-learn machine learning library.
















