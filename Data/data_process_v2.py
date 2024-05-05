import csv
import json
import os
import re

import emoji
from grapheme import graphemes
from langdetect import detect, LangDetectException
from tqdm import tqdm


def emoji_name_to_unicode(path):
    """

    Args:
        path: file path containing the raw data files

    Returns: unicode of 43 emojis based on the raw data files names

    """
    # List files in the specified directory
    files = os.listdir(path)

    # Initialize a list to hold the Unicode emojis
    unicode_emojis = set()

    # Loop through each file name
    for file in files:
        if file.endswith(".csv"):
            # Remove the '.csv' and replace underscores with spaces
            emoji_name = file[:-4]

            try:
                # Convert emoji name to its Unicode character
                unicode_char = emoji.emojize(f":{emoji_name}:")
                unicode_emojis.add(unicode_char.encode('unicode-escape').decode('utf-8').lower())
            except KeyError:
                print(f"No emoji found for: {emoji_name}")
    return unicode_emojis


def get_all_emoji_unicode(input_file: str):
    """

    Args:
        input_file: file containing all emoji unicode

    Returns: processed unicode set

    """
    all_emoji = set()
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            unicodes = line.split()
            processed_unicode = ''
            for unicode in unicodes:
                if len(unicode) == 7:
                    processed_unicode += '\\' + unicode.replace('+', '000')
                else:
                    processed_unicode += '\\' + unicode.replace('+', '')

            all_emoji.add(processed_unicode.lower())
    return all_emoji


def process_emoji(file, output_file, emoji_set, repeat_emoji=False):
    """

    Args:
        file: raw data file to be processed
        output_file: name of the output file
        repeat_emoji: boolean value to decide whether the emoji list should contain repeated emojis
        emoji_set: set containing emoji unicode to filter and process the data
    Returns:
        total number of tweets in the input file
    """
    data = {}
    with open(file, 'r', encoding='utf-8') as f:
        with open(output_file, 'w', newline='', encoding='utf-8') as out_file:
            csv_writer = csv.writer(out_file)
            csv_writer.writerow(['id', 'Comments', 'raw text', 'emojis'])
            csv_reader = csv.reader(f)
            next(csv_reader)  # skip the header line
            id = 0
            for item in tqdm(csv_reader):
                if len(item) == 0:  # skip empty lines
                    continue
                comment = remove_mentions_and_links(item[0])  # strip off the web link and twetter handle
                plain_text = ''
                emoji_list = []
                item = ' '.join(comment.split())  # strip the empty spaces
                try:
                    # filter out foreign language
                    language = detect(item)
                    if language == 'en':
                        for cluster in graphemes(item):
                            text = cluster.encode('unicode-escape').decode('utf-8').lower()
                            if text.startswith('\\u') and text in emoji_set:
                                # text = item.encode('unicode-escape').decode('utf-8')
                                # emoji_list.append(text.encode('utf-8').decode('unicode-escape'))
                                emoji_list.append(cluster)
                            else:
                                if text.startswith('\\u'):
                                    continue
                                else:
                                    plain_text += cluster
                        if not repeat_emoji:  # remove repeated emojis
                            emoji_list = list(set(emoji_list))
                        if len(emoji_list) >= 1:
                            data[id] = {
                                'raw text': item,
                                'Plain text': plain_text,
                                'Emojis': emoji_list
                            }
                            id += 1
                except LangDetectException as e:
                    continue
                # print(plain_text, emoji_list)
                # csv_writer.writerow([id, item, plain_text, emoji_list])
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4)
        return id


def remove_mentions_and_links(comment: str):
    cleaned_text = re.sub(r'@\w+', '', comment)  # Remove Twitter handles
    cleaned_text = re.sub(r'https?://\S+', '', cleaned_text)  # Remove links
    return cleaned_text.strip().lower()


if __name__ == '__main__':
    directory = 'archive'
    emoji_info = {}
    # turn the emoji unicode code points into unicode sets
    all_emojis = get_all_emoji_unicode('emoji_unicode_code_points.txt')
    # get 43 emojis from the raw data files
    file_name_emojis = emoji_name_to_unicode('archive')
    filter_emoji_set = all_emojis - file_name_emojis

    for filename in os.listdir(directory):
        print(filename)
        emoji_name = filename.replace('.csv', '')
        emoji_info[emoji_name] = process_emoji("archive/" + filename, "processed_data/json_file/" +
                                               emoji_name + '.json', file_name_emojis)

    # ################ UNCOMMENT BELOW TO GATHER EMOJI INFO #########################
    # with open('emoji_info.csv', 'w', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     # Write header
    #     writer.writerow(['emoji name', 'num of tweeter'])
    #     # Write data
    #     for key, value in emoji_info.items():
    #         writer.writerow([key, value])


