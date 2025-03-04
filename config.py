import os
import torch
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

class Config:
    VISUALIZATION_DIR = "../metrics scripts/visualizations"
    TREEMAP_DIR = os.path.join(VISUALIZATION_DIR, "culture_treemaps")
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    os.makedirs(TREEMAP_DIR, exist_ok=True)

    #because I'm emoji obsessed
    #also, this is how you get emojis to appear on matplotlib for windows
    plt.rcParams["font.family"] = "Segoe UI Emoji"

    #these were for when I did my first runthrough of emotion analysis
    #and I used one trained on twitter data
    #so the program assumed every story was angry.
    emotion_colors = {
        "anger": "#C53030",
        "joy": "#FBBF24",
        "fear": "#8AA2B3",
        "sadness": "#4299E1",
        "disgust": "#68D391",
        "surprise": "#F6E05E",
        "love": "#ED64A6",
        "tie": plt.cm.Pastel1(np.arange(0, 1, 0.1))
    }

    continent_colors = {
        "Africa": "#E6194B",
        "Asia": "#3CB44B",
        "Australia": "#FFE119",
        "Europe": "#0082C8",
        "North America": "#F58231",
        "Pacific": "#911EB4",
        "South America": "#46F0F0",
        "Unknown": "#A9A9A9"
    }
    BASE_DIR = "data"
    DATA_PATH = os.path.join(BASE_DIR, "stories")
    FLORA_FAUNA_PATH = os.path.join(BASE_DIR, "flora_fauna")
    PICKLE_DIR = "pickles"
    LOG_DIR = "logs"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SENTIMENT_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
    EMOTION_MODEL_NAME = "princeton-nlp/sup-simcse-roberta-large"
    USE_EMBEDDINGS = True

    PICKLE_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    MAX_SEQUENCE_LENGTH = 512

    @classmethod
    def get_pickle_path(cls, name):
        if not os.path.exists(cls.PICKLE_DIR):
            os.makedirs(cls.PICKLE_DIR)
        return os.path.join(cls.PICKLE_DIR, f"{name}_{cls.PICKLE_TIMESTAMP}.pkl")

    @classmethod
    def initialize_logging(cls):
        if not os.path.exists(cls.LOG_DIR):
            os.makedirs(cls.LOG_DIR)
        return os.path.join(cls.LOG_DIR, "take3log.jsonl")

JSONL_FILE = Config.initialize_logging()
