# iowaHomes

This is an application built for the Ames Housing Dataset that predicts the price of a house based on the given features with an average accuracy of 87% (measured on test data) using 4 different regression models which are then combined and averaged out to give the final prediction.


It automatically selects the best features from the dataset and saves them out along with the models. 
Afterwards, only the picked features are displayed to the user on the front end website.
If at any point the model is retrained and new features are available as an input from the user, the website will automatically update
and show those features.

## Sources
[https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard]

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Run these commands from the terminal to install the prerequisites
```
sudo apt update
sudo apt-get install python3
sudo apt-get install python-pip3
sudo apt-get install python3-venv
sudo apt-get install net-tools
```

### Installing

First clone the repo by cd'ing into a directory where you want the project to be
e.g /home/user/projects/

Then run
```
git clone https://github.com/x-Xymos/IowaHomes.git
```
cd into the main directory
```
cd IowaHomes/
```
if not already, make the main scripts executable

```
sudo chmod +x setup run_server kill_server
```
run the setup script to create the python virtual environment and install the required dependencies
```
./setup
```
start the server with
```
./run_server
```
If you close the server and get this error when you try to start the server again
```diff
- Error: That port is already in use.
```
then run
```
./kill_server
./run_server
```
## Opening the application
After the server is ran, we can acess the application through the website
```
http://0.0.0.0:8000/predict/estimate/
```

## Running the tests

The tests can be ran using the test.py file inside IowaHomes/automated_tests

The automated tests consist of 2 tests:
score_models - loads in the saved models and tests their accuracy, should run with no errors and give a a fairly good accuracy score i.e RMSLE - 0.0001 Variance 0.87
run_predictions - this runs the run_predictions function 15 times, passing in random features and values every iteration, this should complete with no errors


## Retraining the model
If at any point the model needs to be retrained because of new data,you can run this function:
```
IowaHomes/iowaHomes/predict/predictModel/save_models_to_file.py
```
Upon successfull completion the terminal should display the filenames of the saved models.
The retrained model can then be tested using the score_models.py function inside the same folder to measure the accuracy.


## Deployment
Deployment to a cloud server or anywhere else is the same as above.
If you want the server to run as a background process then run
```
nohup ./run_server &
```

## Built With

* [Django]
* [Scikit-learn]
* [Sqlite]
