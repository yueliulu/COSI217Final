import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import words

# Download the stopwords corpus if you haven't already
nltk.download('stopwords')
nltk.download("words")
nltk.download("punkt")
english_stop_words = set(stopwords.words("english"))
valid_words = set(words.words())

class Vocab:
    def __init__(self, text, emoji):
        self.text = text
        self.emoji = emoji


def insert_data(d, conn, c):
    with conn:
        c.execute("insert into emoji_counts values (:vocab, :emoji, :cnt)",
                  {'vocab': d.text, 'emoji': d.emoji, 'cnt': 1})


def update_data(vocab, emoji, original_count, conn, c):
    c.execute("UPDATE emoji_counts SET cnt=? WHERE vocab=? AND emoji=?", (original_count + 1, vocab, emoji))
    conn.commit()


def filter_stopwords(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in english_stop_words and word.lower() in valid_words]
    return filtered_words


def update_database(text, emojis, conn, c):
    update_inputs_table(text, emojis, conn, c)
    text = filter_stopwords(text)
    for t in text:
        for e in emojis:
            c.execute("SELECT * FROM emoji_counts WHERE vocab = ? AND emoji = ?", (t,e))
            result = c.fetchone()
            if not result:
                d = Vocab(t, e)
                insert_data(d, conn, c)
            else:
                update_data(t, e, result[-1], conn, c)


def fetch_data(table, c):
    c.execute("SELECT * FROM {}".format(table))
    return c.fetchall()


def find_top_10_vocab(emoji, c):
    c.execute("""
        SELECT vocab, cnt
        FROM emoji_counts
        WHERE emoji = ?
        ORDER BY cnt DESC
        LIMIT 10
    """, (emoji,))
    return c.fetchall()


def update_inputs_table(text, emojis, conn, c):
    with conn:
        c.execute("INSERT INTO inputs (text, emojis) VALUES (?, ?)", (text, ''.join(emojis)))
