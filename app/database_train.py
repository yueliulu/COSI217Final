import sqlite3
import csv

from utils_database import update_database

conn = sqlite3.connect('Database.db')
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS emoji_counts(vocab varchar not null,
                                                     emoji varchar not null,
                                                     cnt integer not null,
                                                     primary key(vocab, emoji));""")

c.execute("""CREATE TABLE IF NOT EXISTS inputs(
                id INTEGER PRIMARY KEY,
                text VARCHAR NOT NULL,
                emojis VARCHAR NOT NULL
            );""")


def load_train_files(file_path):
    with open(file_path, 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)
        head = []
        i=0
        for row in csv_reader:
            if i==0:
                head = row

            else:
                text = row[-1]
                emojis = []
                for index in range(len(row)-1):
                    if row[index]=="1":
                        emojis.append(head[index])
                update_database(text, emojis, conn, c)
            i += 1
            print(i)


def main():
    file_path = '../Data/train.csv'
    load_train_files(file_path)


if __name__ == '__main__':
    main()