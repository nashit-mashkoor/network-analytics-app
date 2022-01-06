# network-analytics-app
A streamlit based web app to display the results of dataset exploration, model training, and model predictions for different use cases

## Installation

Python 3.7.11 is required to run code from this repo.

1) Installing dependencies
```console
$ pip install -r requirements.txt
```
2) Running streamlit server
```console
$ streamlit run app.py
```
## Project structure
Following is a high level break down of the important files for better understanding

```console
$ ls
.
├── README.md
├── data                  # <-- Directory with raw and intermediate data
├── notebooks             # <-- Jupyter notebooks
├── requirements.txt      # <-- Requirements file
├── app.py                # <-- Streamlit server
├── gui.pdf               # <-- PDF file for app requirements
└── src                   # <-- Source code containing associated model and data loader classes
```

