# Applied ML Group 9 project 

## Introduction

This project is for the Applied Machine learning course at the Artificial Intelligence faculty. This project is focused on using pictures of dogs and machine learning to predict the emotion of dogs. 

## Prerequisites to collaborate
Make sure you have the following software and tools installed:

- **Pipenv**: Pipenv is what we used for dependency management. This tools enables users to easily create and manage virtual environments. 

- **DVC**: To manage our data and make sure the data we use between collaborators is identical we use Data Version Control. A git-like library that helps keep track of changes to data, as long as you have the dvc key you can download the same data that we used for training our models. This key can be aquired at ones request.

Structure of the repository:
```bash
├───data  # Stores .csv
├───models  # Stores .pkl
├───notebooks  # Contains experimental .ipynbs
├───project_name
│   ├───data  # For data processing, not storing .csv
│   ├───features
│   └───models  # For model creation, not storing .pkl
├───reports
├───tests
│   ├───data
│   ├───features
│   └───models
├───.gitignore
├───.pre-commit-config.yaml
├───main.py
├───train_model.py
├───Pipfile
├───Pipfile.lock
├───README.md
```

