# Hotel Bookings



The data set consists of 75,166 data points collected by [Antonio, de Almeida, and Nunes (2019)](https://doi.org/10.1016/j.dib.2018.11.126) regarding hotel bookings. 

We'll try to predict whether the reservation includes children or not. The reference contains information on the variables in the data and the outcome column (`children`) is encoded as a factor. Apart from the reference, you can email me (`mxkuhn@gmail.com`) if you have other questions about the data. The data are in the `assignment` path of this repo.

The assignment is to use tidymodels to

* Split the data so that 75% going into training and 25% is for testing. 

* Try a few models and feature engineering methods to improve the area under the ROC curve.

* Use some sort of out-of-sample technique (e.g. validation set, cross-validation, bootstrap, etc.) to compare models prior to using the test set. 

* Pick a model and evaluate the test set. 

* [optional] Make a PR into this repo's assignment directory to show you results. 

Remember to set random number seeds to make the results reproducible and use `sessionifo::session_info()` in your code to report your versions. 

