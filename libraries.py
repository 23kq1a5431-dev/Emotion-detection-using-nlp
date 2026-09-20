!pip install pandas
!pip install matplotlib
!pip install seaborn
!pip install pandas scikit-learn nltk

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import nltk

from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score