import unittest
import sqlite3

from utils_database import filter_stopwords, fetch_data, find_top_10_vocab, update_data, update_database, update_inputs_table

class TestDatabase(unittest.TestCase):
    def setUp(self):
        """
        set up the sqlite connect and the cursor for the test, also remove the changes from previous runs of this unit test 
        """

        self.conn = sqlite3.connect('example.db')
        self.c = self.conn.cursor()

        # each time running the unit test will lead to some difference on the database
        # the lines below are to eliminate those changes
        # this way, no matter how many times we run the unit test, it will always pass
        self.conn.execute("DELETE FROM emoji_counts WHERE emoji = '😠'")
        self.conn.execute("DELETE FROM emoji_counts WHERE emoji = '😭' AND vocab = 'cinnamon'")
        self.conn.execute("UPDATE emoji_counts SET cnt = 1 WHERE vocab = 'hate' AND emoji = '😭'")
        self.conn.execute("DELETE FROM inputs WHERE text = 'I hate cinnamon'")
        self.conn.execute("DELETE FROM inputs WHERE text = 'I like you' AND emojis ='🥚' ")


    def test_filter_stopwords(self):
        """
        test the function filter_stopwords()
        given a string of text in English,
        the expected result should be a list of strings that are not stopwords and are proper English word stems
        """

        text = "when the falls in love with a and an I laugh"
        result = ["love", "laugh"]
        self.assertEqual(filter_stopwords(text),result)
        
        print("filter passed")

    def test_fetch_data(self):
        """
        test the function fetch_data()
        the expected output should have all rows of the requested table
        for a more concise test, here we just need to make sure the output length is correct
        """

        # fetch data from Table emoji_counts
        result1 = fetch_data('emoji_counts', self.c)
        self.assertEqual(len(result1), 2)

        # fetch data from Table inputs
        result2 = fetch_data('inputs', self.c)
        self.assertEqual(len(result2), 1)
        print("fetch passed")

    def test_update_data(self):
        """
        test the function update_data()
        given a word and an emoji, their count in the database (i.e., Column cnt) should be updated
        this function does not return anything but there are changes on the database
        """

        update_data("hate", '😭', 1, self.conn, self.c)

        # make sure the change on cnt is reflected on the database
        updated = self.conn.execute("SELECT cnt FROM emoji_counts WHERE emoji = '😭' AND vocab = 'hate'")
        updated = updated.fetchone()[0]
        self.assertEqual(updated, 2)

        # reset the cnt to the orignal number
        update_data("hate", '😭', updated-2, self.conn, self.c)

        print("update row passed")

    def test_find_top_10_vocab(self):
        """
        test the function find_top_10_vocab()
        given an emoji, the expected result should contain a list of (word, count) tuples
        """
        result3 = find_top_10_vocab('😭',self.c)
        self.assertEqual([('hate', 1), ('Monday', 1)], result3)

        print("top 10 passed")

    def test_update_database(self):
        """
        test the function update_database()
        given a string of plain text and a string of emojis,
        the database should store them as a row in Table inputs,
        and each token should be stored together with each emoji in Table emoji_counts
        this function does not return anything but there are changes on the database
        """

        update_database("I hate cinnamon", "😭😠", self.conn, self.c)

        # make sure the changes in both tables are reflected on the database
        updated1 = self.conn.execute("SELECT * FROM inputs WHERE text = 'I hate cinnamon'")
        updated1 = updated1.fetchone()[2]
        updated2 = self.conn.execute("SELECT * FROM emoji_counts")
        updated2 = updated2.fetchall()

        self.assertEqual(updated1,"😭😠" )
        self.assertEqual(len(updated2), 5)

        print("update db passed")

    def test_update_inputs_table(self):
        """
        test the funciton update_inputs_table()
        given a string of text and a string of emoji, 
        the database should them as a row in Table inputs 
        this function does not return anything but there are changes on the database
        """
        update_inputs_table("I like you", "🥚", self.conn, self.c)

        # make sure the changes in both tables are reflected on the database
        updated3 = self.conn.execute("SELECT * FROM inputs WHERE text = 'I like you'")
        updated3 = updated3.fetchone()[2]

        self.assertEqual(updated3, "🥚")

        print("update inputs passed")




