# MovieClassification
A web based tool using machine learning to determine the IMDb rating given a movie's features.

The approach to this project was as follows:

1.Choosing a dataset 

2.Data preprocessing(Feature selection,dimesion reduction ec.) 

3.Determining the inputs & outputs of the network

4.Feature clustering to determine the most relevant output labels

5.Creation of a multi layer neural network

6.Training the network

7.Testing the network

8.Predicting the rating for unseen data

9.Data visualization

10.Creation of a web page to display results

## Choosing a Dataset

We found an ideal dataset for our purpose on a famous website for datasets which is Kaggle.com.The link to the dataset is as follows:https://www.kaggle.com/deepmatrix/imdb-5000-movie-dataset.

Brief introduction about the dataset

The creator(chuansun76) of this datset had scraped 5000+ movies from IMDB website using a Python library called "scrapy".After the web scapping this datset consisted of 28 variables(or columns) for 5043 movies and 4906 posters (998MB), spanning across 100 years in 66 countries. There are 2399 unique director names, and thousands of actors/actresses. Below are the 28 variables:

"movie_title" "color" "num_critic_for_reviews" "movie_facebook_likes" "duration" "director_name" "director_facebook_likes" "actor_3_name" "actor_3_facebook_likes" "actor_2_name" "actor_2_facebook_likes" "actor_1_name" "actor_1_facebook_likes" "gross" "genres" "num_voted_users" "cast_total_facebook_likes" "facenumber_in_poster" "plot_keywords" "movie_imdb_link" "num_user_for_reviews" "language" "country" "content_rating" "budget" "title_year" "imdb_score" "aspect_ratio"

## Data preprocessing

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

## Designing of the neural network(Supervised Learning Technique)

We chose to create a multi-layer neural network.So first we declared the hyper parameters which were training epoches, number of neurons in the hidden layer one & two and the learning rate which were 2000, 100, 50 and 0.01.Then we declared two placeholders(X & Y) to facilitate the input process of features and targets during runtime. Then each layer's weights and biases were initialized to random values. The activation functions on the hidden layers were sigmoid(which gives an output between 0 & 1) and the final ouput layer had the softmax function(outputs a probabilistic value).

## Training the network

We use 2000 training epoches. when we input a example then make adjustments weights and biases in such a way that the cost will reduce in the next example compared to the current cost. when this is done on the entire training set we say that one epoch is complete.We used the cross entropy function to calculate the cost at each epoch and gradient descent optimzer as our optimization function.This optimizer makes sure the cost function is minimized during each epoch of the learning process by a fraction of the learning rate. 

## Testing the network

The correct predictions are determined by checking the output provided by the network with the expected output on testing data. Then the accuracy is computed as a propotion of correct predictions over the entire number of test data.

## Predicting the rating for an unseen data

We managed to achive an accuracy of 94% on predictions.The graphs that display the accuracy at each epoch is attached above.

## Data visualization

We used Tablue for our vizualization tasks.We wanted to check the spread of the sum of features with their pertaining labels.the resulting visualization diagram is attached above.

## Creation of a webpage to display results

Finally we wanted to develop a user interface to make the application more intutive so we used HTML,CSS,Bootstrap to create a web based application.Python's Flask framewrok(an implementation of the web frameworks concept) was used to our python source code and the web application.

**_Application can be operated as follows:_**

First run the file called Interface.py (This will start the Flask implemented server)
Afterwards the index.html page should be opened via a web browser
And by pressing the button Launch model the result shows the prediction and actual rating respectively.

_**File descriptions**_

**.gitignore**-This file can be used to specify the files which should be ignoredwhen syncing the local repo with the original repo.

**Accuracy&CostPlotting.png**-This plots the graphs of evaluation metrices using TensorBoard visualization tool.

**Cleaning.py**-This is a python script written to extract rows with no missing values or in otherwords eliminate rows with atleast one missing value.

**ComputationalGraph.png**-This image depicts the logigal structure in which tensorflow creates a machine learning model.

**ExampleBiasesPlotting.png**-Depicts the change in the bias distribution for each epoch in the first layer of the neural net.Here each epoch is represented as a slice in the 3d space.

**ExampleWeightsPlotting.png**-Depicts the change in the weight distribution for each epoch in the first layer.Here each epoch is represented as a slice in the 3d space.

**flaskServerScript.py**-This is the script that is used to build a server to communicate with the client(in this scenario localhost) and pass data from the front end to the backend and vice versa.

**index.html**-main static html web page provided to the user to interact with.

**LICENSE**-MIT license

**Movie.py**-This is the file that holds the neural network implementation including training evaluation and saving of the model.

**MovieClustering.py**-This holds the clustering algorithm used to label the dataset using clustering.

**MovieTest.csv**-This is the file that contains the test data set.

**MovieTrain.csv**-This is the file that contains the train data set.

**RatingPredictionWebInterface.png**-This shows a sceernshot of how the results will look for a given input.

**README.md**-This file specifies how the problem was solved and description of the contents of each file in the repository you are in.

**Visualization(Tablue).png**-Visualization done using Tablue on the labels.






 
 

















