## Traffic Sign Recognition Writeup** 

---

## Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[histogram]: ./images/histogram.png "histogram of traffic signs"
[image2]: ./images/grayscale.jpg "Grayscaling"
[image3]: ./images/random_noise.jpg "Random Noise"
[image4]: ./images/image0.jpg "Traffic Sign 1"
[image5]: ./images/image1.jpg "Traffic Sign 2"
[image6]: ./images/image2.jpg "Traffic Sign 3"
[image7]: ./images/image3.jpg "Traffic Sign 4"
[image8]: ./images/image4.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is (31367, 32, 32, 3)
* The size of test set is (12630, 32, 32, 3)
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a histogram chart to show how many tsr in each category in the training set

![alt text][histogram]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

Firstly, I tried to train a model without any preprocess of the image. And then I tried to convert the images into gray scale and normalize the images. I compared the results of different pre-processing and found that after converting the images into gray scale and normalization, the test accuracy improved a little.

I guess that gray scale images could tackle different illumination conditions.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]


#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the first code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by sklearn.model_selection.train_test_split function. The ratio of validation data set is set to 0.2 according to the 20% rule.

My final training set had 31367 number of images. My validation set and test set had 7842 and 12630 number of images.

The size of test data is almost 1/3 of the training data set. More training data should be add. But right now, I have no idea on how to generate more training data. 


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the fifth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray scale image   			    	| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  |                                               | 
| Max polling           | 2x2 stride, outputs 5x5x16                    |
| Flatten       		| flat 5x5x16 to 400 							|
| Fully connected		| 400 to 120									|
| RELU  				|												|
| Dropout               | keep_prob=0.5 for training                    |
| Fully connected       | 120 to 84                                     |
| RELU                  |                                               |
| Fully connected       | 84 to 43                                      |
 


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the sixth cell of the ipython notebook. 

To train the model, I used the LeNet as a template and modify the super paramters of the LeNet.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the seventh cell of the Ipython notebook.

My final model results were:

* training set accuracy of 0.991
* validation set accuracy of 0.975 
* test set accuracy of 0.901

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?
The LenNet architecture was tried.
* What were some problems with the initial architecture?
The accuracy of on test data set is much lower than the accuracy of training and validation data set.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
The test accuracy is lower than validation accuracy, the model is over fitting. I added dropout to the model. Although I am not sure where should dropout be added, dropout did improve the accuracy a little.
* Which parameters were tuned? How were they adjusted and why?
The EPOCH, BATCH_SIZE and learning rate are tuned. I tried to tune the parameters by add and subtract the parameters. The test accuracy is used to identify which parameters are better.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Firstly, the images are trained without any pre-processing. Then, the images were pre-processed by gray-scaling and normalization. The accuracy suggested that the pre-processing steps are needed. Dropout is also needed. The size of training data set is not large enough. The training accuracy is almost 1.0. However, the test accuracy is 0.8. To avoid over-fitting, dropout was utilized.

If a well known architecture was chosen:
* What architecture was chosen?
The LeNet architecture was chosen.
* Why did you believe it would be relevant to the traffic sign application?
The same as handwriting characters, some parts of the traffic signs images are of different color with surrounding pixels. Just as the LeNet could be successfully applied to MNIST data set, it may also work on traffic signs. 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
Both the training and validation accuracy is almost 1.0. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The second image might be difficult to classify because the part of the traffic is too small to identify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| 80 km/h     			| 80 km/h										|
| Stop Sign				| Stop Sign										|
| 30 km/h         		| 30 km/h               		 				|
| 80 km/h       		| 80 km/h             							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00       	| Stop sign   									| 
| 1.39337195e-10    	| No vehicles 									|
| 1.66675809e-13		| Turn right ahead								|
| 1.27382534e-13 		| Keep right		    		 				|
| 6.58082123e-14	    | Priority road     				    		|   

For the second image, the model is relatively sure that this is a 80km/h sign (probability of 0.998), and the image does contain a 80 km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.97974813e-01  		| 80 km/h   									| 
| 1.29839894e-03  		| 100 km/h 										|
| 6.50078873e-04 		| 60 km/h										|
| 6.39253412e-05 		| 50 km/h		    			 				|
| 6.53149345e-06	    | no passing for vehicles over 3.5 tons    		|   
       
For the third image, the model is relatively sure that this is a stop sign (probability of 0.999), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99997258e-01   		| Stop sign   									| 
| 1.59474348e-06   		| turn right ahead								|
| 7.51257346e-07		| keep right									|
| 4.45236850e-07		| no vehicles	    			 				|
| 1.52479795e-09	    | turn left ahead     				    		|   

For the fourth image, the model is relatively sure that this is a 30km/h sign (probability of 0.835), and the image does contain a 30km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 8.35927963e-01   		| 30 km/h   									| 
| 1.63407460e-01   		| 50 km/h										|
| 6.28069160e-04		| 100 km/h										|
| 2.73283295e-05		| 20 km/h		    			 				|
| 4.79346045e-06	    | 80 km/h      	    				    		|   

For the fifth image, the model is relatively sure that this is a 80km/h sign (probability of 0.999), and the image does contain a 80km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99826252e-01    	| 80 km/h   									| 
| 1.07589425e-04    	| 50 km/h 										|
| 4.82312753e-05 		| 60 km/h										|
| 1.24513399e-05		| 100 km/h		    			 				|
| 5.48849721e-06	    | 30 km/h      	    				    		|   
       
    