import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download the stopwords corpus if you haven't already
nltk.download('stopwords')
nltk.download('punkt')
english_stop_words = stopwords.words('english')


class Vocab:
    def __init__(self, text, emoji):
        self.text = text
        self.emoji = emoji


def insert_data(conn, c, d):
    with conn:
        c.execute("insert into emoji_counts values (:vocab, :emoji, :cnt)",
                  {'vocab': d.text, 'emoji': d.emoji, 'cnt': 1})


def update_data(conn, c, vocab, emoji, original_count):
    c.execute("UPDATE emoji_counts SET cnt=? WHERE vocab=? AND emoji=?", (original_count + 1, vocab, emoji))
    conn.commit()


def filter_stopwords(text):
    words = word_tokenize(text)
    stop_words = set(english_stop_words)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered = ' '.join(filtered_words)
    return filtered
