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

## How to run the app
After clone the current directory, go to https://drive.google.com/drive/folders/1EFLz33tn4HC4Dwm4CB5qyY7rbjluCy0x, download best_model.pt, and move it into
the "model" folder. 

With the .pt file in place, choose one of the following ways to run:

### A. Run With Docker (requires 16+GB RAM)
First make sure to start the Docker, then use the command below to run the app. Requirements are listed in requirement.txt in the main directory.

```bash
docker build -t app_streamlit .
docker run -dp 8501:8501 app_streamlit
```
Then you can access the streamlit application at: http://localhost:8501/

Given the large size of our Database.db, running the app with Docker would require the computer with RAM larger than 16gb. If the RAM is not large enough, the app might disconnect with no error after several requests.

### B. Run without Docker

To run the app without Docker, comment out line 5 and 11 in app_streamlit.py, and comment in line 6, 7, and 12 of the same file. Navigates to app folder then use the command below to run the app:

```bash
streamlit run app_streamlit.py
```

## How to run the tests 
The test results can only be seen on the IDE. 

To run the unit test for data cleaning, please go to the "Data" folder and run data_cleaning_unit_test.py file on the IDE 

To run the unit test for database, please go to the "app" folder and run db_unit_test.py on the IDE 
