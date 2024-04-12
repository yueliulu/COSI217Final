import csv
import html
import os
import re

from collections import Counter
import random

# import emoji


# TODO
# remove tweets only have a single emoji or the same emoji multiple times
# replace html characters
# include multiple emojis tweets

def process_emoji(input_file: str, output_file: str):
    """

    Args:
        input_file: original file name contains unprocessed comments and emojis
        output_file: csv file with id and processed comments that only contain 1 type of emoji per file

    Returns: None

    """
    with open(input_file, 'r', encoding='utf-8') as f:
        with open(output_file, 'w', newline='', encoding='utf-8') as out_file:
            csv_writer = csv.writer(out_file)
            csv_writer.writerow(['ID', 'Comments'])
            csv_reader = csv.reader(f)
            # next(csv_reader)
            id = 0
            # ------- uncomment the following if you want only the dominate emoji in your output data ------ #
            # dominate_emoji = ':' + emoji_name + ':'
            # dominate_emoji_character = emoji.emojize(dominate_emoji)
            for item in csv_reader:
                if len(item) == 0:  # skip empty lines
                    continue
                item = item[0]
                item_l = item.split()
                # strip off the web link and twetter handle
                comment_list = [i for i in item_l if "@" not in i and 'https' not in i]
                comment = ' '.join(comment_list)
                if not remove_only_emojis(comment):
                    id += 1
                    commtent = remove_html_tags(comment)
                    csv_writer.writerow([id, commtent])

                # character_counter = Counter(comment)
                # emojis = {}
                # for char, count in character_counter.items():
                #     if emoji.is_emoji(char):
                #         emojis.update({char: count})
                # if len(emojis) == 1 and emoji.demojize(list(emojis.keys())[0]) == dominate_emoji:
                #     id += 1
                #     csv_writer.writerow([id, comment])


def remove_only_emojis(tweet: str):
    '''

    Args:
        tweet: input tweet comments

    Returns: return True if the tweet only contains emoji and only one kind of emoji


    '''
    emojis = []
    strings = []
    i = 0
    while i < len(tweet):
        # Emojis can take up multiple characters, so we need to handle them accordingly
        if 0x1F000 <= ord(tweet[i]) <= 0x1FFFF:
            emojis.append(tweet[i])
            i += 1
        elif 0x2000 <= ord(tweet[i]) <= 0x3FFF or 0x4000 <= ord(tweet[i]) <= 0xDFFF or 0xE000 <= ord(
                tweet[i]) <= 0xFFFF:
            emojis.append(tweet[i:i + 2])
            i += 2
        else:
            strings.append(tweet[i])
            i += 1
    # return True if the tweet only contains one kind of emoji and no context
    return len(Counter(emojis)) == 1 and len(strings) == 0


def remove_html_tags(comment):
    # Remove HTML tags
    clean_string = re.sub('<[^<]+?>', '', comment)
    # Remove HTML escape characters
    clean_string = re.sub('&[^\s]*;', '', clean_string)
    return clean_string


def pilot_annotation(num: int, input_file: str, annotation_file: str, left_annotation_file: str):
    '''

    Args:
        num: num of tweets to randomly select
        input_file: input tweets file
        annotation_file: file containing randomly selected tweets to be annotated
        left_annotation_file: tweets left

    Returns:

    '''
    to_annotate = []
    left_annotation = []

    # random select num number of tweets
    with open(input_file, 'r', encoding='utf-8') as f:
        csv_reader = list(csv.DictReader(f))
        num_lines = int(csv_reader[-1]["ID"])  # get the total number of tweets in the input file
        random_ids = sorted(random.sample(range(1, num_lines), num))

        # get rows to be annotated and rows left

        for tweet in csv_reader:
            if int(tweet['ID']) in random_ids:
                to_annotate.append(tweet)
            else:
                left_annotation.append(tweet)

    # generating file for annotation
    with open(annotation_file, 'w', newline='', encoding='utf-8') as annotation_file:
        csv_writer = csv.DictWriter(annotation_file, fieldnames=['ID', 'Comments'])
        csv_writer.writeheader()
        csv_writer.writerows(to_annotate)

    # generating file of left annotations
    with open(left_annotation_file, 'w', newline='', encoding='utf-8') as left_annotation_file:
        csv_writer = csv.DictWriter(left_annotation_file, fieldnames=['ID', 'Comments'])
        csv_writer.writeheader()
        csv_writer.writerows(left_annotation)


def get_emojis_info(dir: str, output_file: str):
    """

    """
    with open(output_file, 'w', encoding='utf-8')as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['emoji name', 'num of tweets'])
        for file in os.listdir(dir):
            emoji_name = file.replace(".csv", "")
            csv_reader = list(csv.DictReader(file))
            num_lines = int(csv_reader[-1]["ID"])  # get the total number of tweets in the input file
            csv_writer.writerow([emoji_name, num_lines])


if __name__ == '__main__':
    directory = 'Data/processed_data/'
    emojis = ['face_holding_back_tears(updated).csv', 'smiling_face_with_tear(updated).csv',
              'loudly_crying_face(updated).csv']
    # for emoji in emojis:
    #     pilot_annotation(100, f"{emoji}", f"(to_annotate){emoji}", f'(left){emoji}')
    # process_emoji("Data/archive/face_holding_back_tears.csv", "face_holding_back_tears.csv")
    # process_emoji("Data/archive/smiling_face_with_tear.csv", 'smiling_face_with_tear.csv')
    # process_emoji("Data/archive/loudly_crying_face.csv", 'loudly_crying_face.csv')

    ################## UNCOMMENT BELOW TO PROCESS THE WHOLE FOLDER ##########################
    # for filename in os.listdir(directory):
    #     emoji_name = filename.replace('.csv', '')
    #     process_emoji("Data/archive/" + filename, "Data/processed_data/" + filename)

    get_emojis_info(directory, "emoji_info.csv")