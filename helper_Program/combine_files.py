#### If the files has only text and no numerical if numerical value exist the int(x.split) does not work so used natural sort first
# import os

# # Directory containing the text files
# directories = [
#     "/Users/jibanchaudhary/Documents/Projects/legal_assistance/English/dataset/kaggle_aila_dataset/Object_statutes"
# ]

# for directory in directories:
# # List all files in the directory
#     files = [f for f in os.listdir(directory) if f.endswith('.txt')]

#     # Open the output file in write mode
#     with open('/Users/jibanchaudhary/Documents/Projects/legal_assistance/English/dataset/legal_corpus.txt', 'a', encoding='utf-8') as outfile:
#         # Iterate through each file
#         for filename in sorted(files, key=lambda x: int(x.split('.')[0])):  # Sorting by number if filenames are numbered
#             filepath = os.path.join(directory, filename)
#             # Open and read the content of each file
#             with open(filepath, 'r', encoding='utf-8') as infile:
#                 content = infile.read()
#                 # Write the content to the output file
#                 outfile.write(content)
#                 outfile.write("\n")  # Add a newline character after each file's content for separation


# #######Natural sort
# import os
# import re
# # Directory containing the text files
# directories = "/Users/jibanchaudhary/Documents/Projects/legal_assistance/English/dataset/kaggle_aila_dataset/Object_casedocs"

# # List all .txt files in the directory
# files = [f for f in os.listdir(directories) if f.endswith('.txt')]

# def natural_sort(s):
#      return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

# # Open the output file in append mode
# with open('/Users/jibanchaudhary/Documents/Projects/legal_assistance/English/dataset/kaggle_aila_dataset/object_casedocs.txt', 'a', encoding='utf-8') as outfile:
#     # Iterate through each file, sorted by number if filenames are numbered
#     for filename in sorted(files, key=natural_sort):
#         filepath = os.path.join(directories, filename)
#         # Open and read the content of each file
#         with open(filepath, 'r', encoding='utf-8') as infile:
#             content = infile.read()
#             # Write the content to the output file
#             outfile.write(content)
#             outfile.write("\n")  # Add a newline after each file's content

### Formatting


# import os

# # Define the directory path
# dir_path = "/Users/jibanchaudhary/Documents/Projects/legal_assistance/dataset/kaggle_dataset_1/nepali_news_dataset_20_categories_large/nepali_news_dataset_20_categories_large"

# # Iterate through the directory and list all folder paths
# for root, dirs, files in os.walk(dir_path):
#     for directory in dirs:
#         print(os.path.join(root, directory))


import os

file_path = "/Users/jibanchaudhary/Documents/Projects/legal_assistance/English/dataset/kaggle_aila_dataset/object_casedocs.txt"

# Check if the file exists
if not os.path.isfile(file_path):
    print(f"File not found: {file_path}")
else:
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split the text into sentences based on the full stop followed by a space
    sentences = text.split('. ')

    # Join the sentences with a full stop and newline character
    formatted_sentence = '.\n'.join(sentences).strip() + '.'

    # Write the formatted text back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(formatted_sentence)

    print("Formatting successful")
