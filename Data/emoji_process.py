import csv
import json
import os
import re

from collections import Counter
from langdetect import detect, LangDetectException
from tqdm import tqdm
import random
import emoji
import os


# TODO
# remove tweets only have a single emoji or the same emoji multiple times
# replace html characters
# include multiple emojis tweets

def emoji_name_to_unicode(path):
    # List files in the specified directory
    files = os.listdir(path)

    # Initialize a list to hold the Unicode emojis
    unicode_emojis = []

    # Loop through each file name
    for file in files:
        if file.endswith(".csv"):
            # Remove the '.csv' and replace underscores with spaces
            emoji_name = file[:-4]

            try:
                # Convert emoji name to its Unicode character
                unicode_char = emoji.emojize(f":{emoji_name}:")
                unicode_emojis.append(unicode_char)
            except KeyError:
                print(f"No emoji found for: {emoji_name}")

    return unicode_emojis


def process_emoji(input_file: str, output_file: str, valid_emojis):
    """

    Args:
        input_file: original file name contains unprocessed comments and emojis
        output_file: csv_file with id and processed comments that only contain 1 type of emoji per file

    Returns: None

    """
    data = {}  # Dictionary to store the data
    with open(input_file, 'r', encoding='utf-8') as f:
        # with open(output_file, 'w', newline='', encoding='utf-8') as out_file:
        #     csv_writer = csv.writer(out_file)
        #     csv_writer.writerow(['ID', 'Comments', 'raw text', 'emojis'])
        csv_reader = csv.reader(f)
        next(csv_reader)
        id = 0
        # ------- uncomment the following if you want only the dominate emoji in your output data ------ #
        # dominate_emoji = ':' + emoji_name + ':'
        # dominate_emoji_character = emoji.emojize(dominate_emoji)
        for item in tqdm(csv_reader):
            if len(item) == 0:  # skip empty lines
                continue
            comment = remove_mentions_and_links(item[0])  # strip off the web link and twetter handle
            valid_comment, text_emoji_map = remove_only_emojis(comment)
            if valid_comment:
                id += 1
                comment = remove_html_tags(comment)

                emojis = [emoji for emoji in text_emoji_map[1] if emoji in valid_emojis]
                data[id] = {
                    "Comments": comment,
                    "raw text": text_emoji_map[0],
                    "emojis": emojis
                }
                    # csv_writer.writerow([id, comment, text_emoji_map[0], emojis])
    # Write the dictionary to a JSON file
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)
    return id


def remove_mentions_and_links(comment: str):
    cleaned_text = re.sub(r'@\w+', '', comment)  # Remove Twitter handles
    cleaned_text = re.sub(r'https?://\S+', '', cleaned_text)  # Remove links
    return cleaned_text.strip().lower()


def remove_only_emojis(comment: str):
    """

    Args:
        tweet: input tweet comments

    Returns: return True if the tweet only contains emoji and no context


    """
    unicode_range = r'[\U0001F004-\U0001F9F6\U0001FA70-\U0001FA73\U0001FA78-\U0001FA7A\U0001FA80-\U0001FA82' \
                    r'\U0001FA90-\U0001FA95\U0001FAA0-\U0001FAA8\U0001FAB0-\U0001FAB6\U0001FAC0-\U0001FAC2' \
                    r'\U0001FAD0-\U0001FAD6\U0001FAE0-\U0001FAE7\U0001FAF0-\U0001FAF4]'
    emojis = []
    cleaned_text = ''
    i = 0
    while i < len(comment):
        # Check if the current character is an emoji
        if re.match(unicode_range, comment[i]):
            # Find the end index of the current emoji sequence
            end_index = i + 1
            while end_index < len(comment) and re.match(
                    unicode_range,
                    comment[end_index]):
                end_index += 1
            # Append each emoji character individually to the emojis list
            emojis.extend(list(comment[i:end_index]))
            i = end_index
        else:
            # Append non-emoji characters to the cleaned text
            cleaned_text += comment[i]
            i += 1
    # return True if the tweet only contains emoji and no context or only text no emoji
    try:
        language = detect(cleaned_text)
        if len(emojis) > 0 and len(cleaned_text) > 0 and language == "en":
            return True, (cleaned_text, emojis)
        return False, (cleaned_text, emojis)
    except LangDetectException as e:
        return False, (cleaned_text, emojis)



def remove_html_tags(comment):
    # Remove HTML tags
    clean_string = re.sub('<[^<]+?>', '', comment)
    # Remove HTML escape characters
    clean_string = re.sub('&[^\s]*;', '', clean_string)
    return clean_string


def text_emoji_mapping(comment):
    """
        Args:
            comment: single tweeter comment
        Return:
            tuple containing the raw text and all emojis containing in the tweeter comment
    """


if __name__ == '__main__':
    # directory = 'Data/processed_data/'

    directory = 'Data/archive'
    emojis_list = emoji_name_to_unicode(directory)

    emoji_info = {}
    process_emoji("archive/egg.csv", 'sample.csv', emojis_list)
    # ################# UNCOMMENT BELOW TO PROCESS THE WHOLE FOLDER ##########################
    # for filename in os.listdir(directory):
    #     emoji_name = filename.replace('.csv', '')
    #     emoji_info[emoji_name] = process_emoji("Data/archive/" + filename, "Data/processed_data/csv_file/" +
    #                                            emoji_name + '.csv', valid_emojis=emojis_list)
    #
    # ################ UNCOMMENT BELOW TO GOTHER EMOJI INFO #########################
    # with open('emoji_info.csv', 'w', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     # Write header
    #     writer.writerow(['emoji name', 'num of tweeter'])
    #     # Write data
    #     for key, value in emoji_info.items():
    #         writer.writerow([key, value])
