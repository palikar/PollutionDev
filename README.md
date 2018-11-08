# Creating and Evaluating Stochastic Regression Models on the Basis of Heterogeneous Sensor Networks

This repository contains all the source code, documentation and presentations for my Bachelor&rsquo;s Thesis I wrote in the Summer of 2018.


## Abstract

This thesis aims to better understand Bayesian machine learning models and their practical use on real world data. We examine two models that incorporate uncertainty in their predictions – Bayesian Neural Networks and Mixture Density Networks. The used data comes from air-pollution sensors. The quality of three of the sensors is known to be high but for the rest of them the quality of measurement is unknown. We aim to build a model that can predict the air pollution at some sensor at a given time. Consideration of the uncertainty in the predicted value is crucial as it allows the precise evaluation of the generated models. We compare the models through evaluation with proper scoring rules. As the quality of the majority of sensors is unknown, we try to find out which of the sensors are most relevant for the prediction through a feature importance technique. We leverage the capabilities of Tensorflow, Edward and GPFlow as machine learning libraries in order to build probabilistic regression models that can be further evaluated.


## Thesis

The whole thesis can be found [here](./Thesis/thesis.pdf). If you want to &ldquo;understand&rdquo; &ldquo;everything&rdquo; that I worked on in my thesis you should read the whole PDF file. Everything is covered there. From the motivation, the theory, the way everything is calculated and implemented as well as the results. The PDF file is, however, a little bit lengthy. A much more short version (or summery) of the work is the [transcript](./Final-Presentation/Sayings/final-pres-english.md) of the final presentation that I gave on the thesis. The [slides](./Final-Presentation/Final-Vortrag.pdf) for that are also publicly available. Right before the start of the thesis I also had to hold a presentation about &ldquo;what will be done in the thesis&rdquo;. The [slides](./BA-Vortrag/BA-Vortrag.pdf) from that also public and offer some service level overview of my plans at the start. Some things had changed though.


## Summery
