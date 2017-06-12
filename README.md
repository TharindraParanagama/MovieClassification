# MovieClassification
A web based tool using machine learning to determine the IMDb rating of a given the movie's features.

The approach to this project was as follows:

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

First we chose the most relevant features which we thought would be meaningful features that would directly impact the rating            of a movie.

Feature selection/Inputs 

  num_critic_for_reviews(as # of critics)
  movie_facebook_likes(as Total likes for the movie)
  director_facebook_likes(as Director’s FB likes)
  actor_1_facebook_likes(as Main Actor’s FB likes)
  gross(as Revenue from the movie)

Outputs

We chose three classes which were poor, moderate and excellent.These were denoted by 0,1 and 2 respectively.

One-hot encoding

This is a binary encoding technique to convert categorical values to nominal values.Here we encoded our labels as follows:

    0=[1 0 0]
    1=[0 1 0]
    2=[0 0 1]

We also had to get rid of missing values for which we wrote a code to check whether any of the cells contained missing values if so we eliminated all such records.This reduced the dataset to 4149 records.

Clustering(Unsupervised Learning Technique)

Now we have to determine the labels of the features for that we used K-Means clutering algorithm based on the similarities of the input features.This algorithm was easyly implemented using Google's scikit-learn machine learning library.

Designing of the neural network(Supervised Learning Technique)

We chose to create a multi-layer neural network.So first we declared the hyper parameters which were training epoches, number of neurons in the hidden layer one & two and the learning rate which were 2000, 100, 50 and 0.01.Then we declared two placeholders(X & Y) to facilitate the input process of features and targets during runtime. Then each layer's weights and biases were initialized to random values. The activation functions on the hidden layers were sigmoid(which gives an output between 0 & 1) and the final ouput layer had the softmax function(outputs a probabilistic value).

Training the network

We use 2000 training epoches. when we input a example then make adjustments weights and biases in such a way that the cost will reduce in the next example compared to the current cost. when this is done on the entire training set we say that one epoch is complete.We used the cross entropy function to calculate the cost at each epoch and gradient descent optimzer as our optimization function.This optimizer makes sure the cost function is minimized during each epoch of the learning process by a fraction of the learning rate. 

Testing the network

The correct predictions are determined by checking the output provided by the network with the expected output on testing data. Then the accuracy is computed as a propotion of correct predictions over the entire number of test data.















