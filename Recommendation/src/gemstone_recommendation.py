from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

