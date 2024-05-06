# COSI217 Final Project: Emoji Generator for Texts 🤖🌐🤔

## Data 📄
The Data directory comprises two folders: "archive" and "processed_data". Within the archive folder are 43 CSV files, 
each corresponding to a raw Twitter file containing prevalent emojis, as indicated by the file names. Meanwhile, the 
processed_data folder houses both CSV and JSON formatted datasets, which have been processed and cleaned. The data 
processing was executed using the code found in "data_process_v2.py".

The file data_process_v2.py executes the following data cleaning procedures:

1. Eliminates web links and Twitter handles.
2. Filters out content in foreign languages.
3. Removes tweets that solely consist of emojis without any accompanying context.

All the imports are listed on the top of this file, and the processed data is already stored in "processed_data/json_file".
To execute this code again, import the packages if they haven't been imported already, and then run the code.

File emoji_info.csv provides an overview of the dataset, detailing the number of tweets within each processed file
File emoji_unicode_code_points.txt contains most of the unicode points for the emojis. This file is used to detact emojis 
used in each tweet. 

## How to run 
After clone the current directory, go to https://drive.google.com/drive/folders/1EFLz33tn4HC4Dwm4CB5qyY7rbjluCy0x, download best_model.pt, and move it into
"model" folder. 

With the .pt file in place, go to the "app" directory and run the app:

(bash here)
