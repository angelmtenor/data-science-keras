# Data science projects with Keras

[Angel Martinez-Tenor](https://profile.angelmtenor.com/).

**Disclaimer: This notebooks-based repo was developed in early 2018. Since July 2022, I'm updating it using the best practices I've learned implementing solutions in production environment my experience as a lead data scientist**

This repo contains a set of data science projects solved with artificial neural networks implemented in [Keras](https://github.com/keras-team/keras/). It is based on a set of use cases from [Udacity](https://www.udacity.com/), [Coursera](https://www.coursera.org/) & [Kaggle](https://www.kaggle.com/)

[Github repo](https://github.com/angelmtenor/data-science-keras)
<br>

## Scenarios
### Classification models

- [Enron Scandal](https://github.com/angelmtenor/data-science-keras/blob/master/notebooks/enron_scandal.ipynb) Identifies Enron employees who may have committed fraud

- [Property Maintenance Fines](https://github.com/angelmtenor/data-science-keras/blob/master/notebooks/property_maintenance_fines.ipynb) Predicts the probability of a set of blight tickets to be paid on time

- [Sentiment IMDB](https://github.com/angelmtenor/data-science-keras/blob/master/notebooks/sentiment_IMDB.ipynb)  Predicts positive or negative sentiments from movie reviews (NLP)


- [Spam detector](https://github.com/angelmtenor/data-science-keras/blob/master/notebooks/spam_detector.ipynb) Predicts the probability that a given email is a spam email (NLP)

- [Student Admissions](https://github.com/angelmtenor/data-science-keras/blob/master/notebooks/student_admissions.ipynb)  Predicts student admissions to graduate school at UCLA

- [Titanic](https://github.com/angelmtenor/data-science-keras/blob/master/notebooks/titanic.ipynb)  Predicts survival probabilities from the sinking of the RMS Titanic

### Regression models

- [Bike Rental](https://github.com/angelmtenor/data-science-keras/blob/master/notebooks/bike_rental.ipynb) Predicts daily bike rental ridership

- [House Prices](https://github.com/angelmtenor/data-science-keras/blob/master/notebooks/house_prices.ipynb) Predicts house sales prices from Ames Housing database

- [Simple tickets](https://github.com/angelmtenor/data-science-keras/blob/master/notebooks/simple_tickets.ipynb)  Predicts the number of tickets requested by different clients


### Recurrent models

- [Machine Translation](https://github.com/angelmtenor/data-science-keras/blob/master/notebooks/machine_translation.ipynb)  Translates sentences from English to French (NLP)

- [Simple Stock Prediction](https://github.com/angelmtenor/data-science-keras/blob/master/notebooks/simple_stock_prediction.ipynb) Predicts Alphabet Inc. stock price

- [Text generator](https://github.com/angelmtenor/data-science-keras/blob/master/notebooks/text_generator.ipynb) Creates an English language sequence generator (NLP)

### Social network models

- [Network](https://github.com/angelmtenor/data-science-keras/blob/master/notebooks/network.ipynb)  Predicts missing salaries and new email connections from a company's email network




## Instructions
*Python 3.7+ required. Conda environment with Python 3.10 suggested*


1. Clone the repository using `git`:
    ```
    git clone https://github.com/angelmtenor/data-science-keras.git
    ```

2. Create a virtual/conda environment (optional):
    ```
    conda create -n ds-keras python=3.10
    conda activate ds-keras
    ```

3. In the folder of the cloned repository, install the dependencies (Numpy, Matplotlib, Seaborn, Pillow, TensorFlow, and Keras):
    ```
    cd data-science-keras
    pip install -r requirements.txt
    ```

    To install tensorflow with GPU support, follow the instructions of this guide: [Install TensorFlow GPU](https://www.tensorflow.org/install/pip#install_cuda_with_apt).


4. Open the desired project/s with [Jupyter Notebook](http://jupyter.readthedocs.io/en/latest/install.html)
    ```
    cd data-science-keras
    jupyter notebook
    ```

*Tested on both, pure Ubuntu 22 with no GPU and Ubuntu 22 with RTX 2070 on WSL (Windows 11)*
