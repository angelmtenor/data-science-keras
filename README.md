# Data science projects with Keras (Poetry Version)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)<br>

Author: [Angel Martinez-Tenor](https://profile.angelmtenor.com/)

Repository: [Github link](https://github.com/angelmtenor/data-science-keras)


This repo contains a set of data science projects solved with artificial neural networks implemented in [Keras](https://github.com/keras-team/keras/). It is based on a set of use cases from [Udacity](https://www.udacity.com/), [Coursera](https://www.coursera.org/) & [Kaggle](https://www.kaggle.com/)

The repo also introduces a minimal package **ds_boost** initally implemented as a helper for this repo




Disclaimer: This notebooks-based repo was developed in early 2018. Since July 2022, I'm updating it using the best practices I've learned implementing solutions in production environment my experience as a lead data scientist


A **non-poetry version** of this repo is available in the branch `no-poetry`

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


## Setup & Usage
*Python 3.8+ required. Conda environment with Python 3.10 suggested*

1. Clone the repository using `git`:

    ```bash
    git clone https://github.com/angelmtenor/data-science-keras.git
    ```

2. Enter to the root path of the repo and use or create a new conda environment for development:

```bash
$ conda create -n dev python=3.10 -y && conda activate dev
```

3. Install the minimal package developed as a helper for this repo:
    ```bash
    pip install dist/ds_boost-0.1.0-py3-none-any.whl
    ```

4. Open the desired project/s with [Jupyter Notebook](http://jupyter.readthedocs.io/en/latest/install.html)
    ```bash
    cd data-science-keras
    jupyter notebook
    ```

## Development Mode
In the root folder of the cloned repository, install all the required dev packages and the ds-boost mini package (**Make** required):
```bash
make setup
```

To install tensorflow with GPU support, follow the instructions of this guide: [Install TensorFlow GPU](https://www.tensorflow.org/install/pip#install_cuda_with_apt).

QA (manual pre-commit):
```bash
make qa
```

###  Development Tools Required:

**A Container/Machine with Conda, Git and Poetry as closely as defined in `.devcontainer/Dockerfile`:**

- This Dockerfile contains a non-root user so the same configuration can be applied to a WSL Ubuntu Machine and any Debian/Ubuntu CLoud Machine (Vertex AI workbench, Azure VM ...)
- In case of having an Ubuntu/Debian machine with non-root user (e.g.: Ubuntu in WSL, Vertex AI VM ...), just install the tools from  *non-root user (no sudo)** section of `.devcontainer/Dockerfile`  (sudo apt-get install \<software\> may be required)
- A pre-configured Cloud VM usually has Git and Conda pre-installed, those steps can be skipped
- The development container defined in `.devcontainer/Dockerfile` can be directly used for a fast setup (Docker required).  With Visual Studio Code, just open the root folder of this repo, press `F1` and select the option **Dev Containers: Open Workspace in Container**. The container will open the same workspace after the Docker Image is built.


## Contributing

Check out the contributing guidelines

## License

`ds_boost` was created by Angel Martinez-Tenor. It is licensed under the terms of the MIT license.

## Credits

`ds_boost` was created from a Data Science Template developed by Angel Martinez-Tenor. The template was built upon `py-pkgs-cookiecutter` [template] (https://github.com/py-pkgs/py-pkgs-cookiecutter)
