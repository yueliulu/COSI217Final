import unittest
from data_process_v2 import get_all_emoji_unicode, remove_mentions_and_links
from emoji_process  import remove_only_emojis

class TestDataProcess(unittest.TestCase):
    def test_get_all_emoji_unicode(self):
        input_file = "./emoji_unicode_code_points.txt"
        result_list = get_all_emoji_unicode(input_file)

        # correct result for U+1F507
        output_simple_emoji = '\\u0001f507'

        # correct result for U+1F9D1 U+200D U+1F4BB
        output_complex_emoji = '\\u0001f9d1\\u200d\\u0001f4bb'

        self.assertIn(output_simple_emoji, result_list)
        self.assertIn(output_complex_emoji, result_list)

    def test_remove_mentions_and_links(self):
        comment = "@JohnDoe234 My coach said I should try this link: https://ghotimeansfish.com"
        output = "my coach said i should try this link:"
        self.assertEqual(remove_mentions_and_links(comment), output)

    def test_remove_only_emojis(self):
        comment_foreign_language = "チェンさん、お疲れ様でした🙂"
        comment_only_emojis = "🫒🙅🍣🤔"
        comment_only_text = "they understood the assignment"
        comment_standard = "oh yeah sure 🔥"
        boolean1, tuple1 = remove_only_emojis(comment_foreign_language)
        self.assertFalse(boolean1)
        self.assertIn(["🙂"], tuple1)
        self.assertIn('チェンさん、お疲れ様でした', tuple1)
        boolean2, tuple2 = remove_only_emojis(comment_only_emojis)
        self.assertFalse(boolean2)
        self.assertIn(['🫒', '🙅', '🍣', '🤔'], tuple2)
        self.assertIn('', tuple2)
        boolean3, tuple3 = remove_only_emojis(comment_only_text)
        self.assertFalse(boolean3)
        self.assertIn([],tuple3)
        self.assertIn("they understood the assignment", tuple3)
        boolean4, tuple4 = remove_only_emojis(comment_standard)
        self.assertTrue(boolean4)
        self.assertIn(["🔥"], tuple4)
        self.assertIn("oh yeah sure ", tuple4)




