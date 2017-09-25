# ML-vs-DS
Find keywords in text relative to control text

## Project Goal

This is a sequence of scripts used to find keywords, and compare job descriptions for Machine Learning and Data Science jobs. Job descriptions were scraped from the web, and are not included here (to avoid any possible copywrite violations). A more detailed project description is in the file Project-Description.pdf.

## Finding keywords

Functions for finding keywords are found in process-text.jl, plotting functions in graph-word-norm.jl, and a script for using these is in process-script.jl.
The user must write loadDesc.jl with a julia function loadDesc that loads documents into 3 lists of strings, the first of which contains the control documents, and the next two the sets of documents to be compared.

## Comparing Documents

A Word2Vec model and an SVM are written in python (with TensorFlow for Word2Vec) in word2vec.py
