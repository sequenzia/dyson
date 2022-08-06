# Dyson: Modeling/Algorithm Research & Development 
An evolving collection of data models and algorithms used to model financial markets including equities, options, futures and crypto assets. The project has grown from only traditional statistical modeling to Machine Learning based modeling which includes deep neural networks and probabilistic reasoning networks.

**Frameworks & Libraries:** TensorFlow, Keras, Pandas, NumPy, Seaborn, Matplotlib, Statsmodels, Scikit-Learn

**Algorithms & Models:** CNNs, RNNs, Transformers, Attention Mechanisms, Temporal Conv Networks, Autoencoders, Bayesian Neural Networks, ARIMA, Structural Time Series

## How it works
All the networks & models are located in [ml_research](/ml_research). They are designed to work with [Photon ML](https://github.com/sequenzia/photon) which subclasses TensorFlow (Keras models).

At the root of the `ml_research` folder there is a `run.py` file that instantiates the framework and network of models. This run file loads a `config.py` file that loads the configurations for the entire network and all the models.

The config file loads [datasets](/ml_research/data) that are in Apache Parquet format.

The config files also loads [models](/ml_research/models) & [layers](/ml_research/layers) that are subclassed from Photon ML which is also a subclass of TensorFlow.

## Run on Google Colab

https://github.com/sequenzia/dyson/blob/master/run_photon.ipynb

*** make sure the Colab Notebook has a GPU runtime type

## Sample Datasets
Sample Market Data: (Apache Arrow Parquet File)

- SPY ETF market data in 1M resolution (2 years: 2016-2017)
- Includes some predefined features that are based of off some standard technical price indicators 
- Includes predefined label groups (1,2,3,4,5,6 & 7 days in the future)
- Discrete price movements with rate of change; used for regression inferences 
- 5 predefined classes for each label group; for used for classification inferences

