import pandas as pd

csv_file_path = "/Users/jibanchaudhary/Documents/Projects/legal_assistance/dataset/clean_date_categories.csv"

chunk_size = 10000

chunks = pd.read_csv(csv_file_path, chunksize=chunk_size)


for chunk in chunks:
    print(chunk.head)
