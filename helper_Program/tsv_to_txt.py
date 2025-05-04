import csv

tsv_file_path = "/Users/jibanchaudhary/Documents/Projects/legal_assistance/dataset/asr_nepali/utt_spk_text.tsv"
existing_file_path = "/Users/jibanchaudhary/Documents/Projects/legal_assistance/dataset/ne_np_female/nepali_corpus.txt"

# Step 1: Read the new sentences from the TSV file
sentences = []
with open(tsv_file_path, mode="r", encoding="utf-8") as file:
    tsv_reader = csv.reader(file, delimiter="\t")
    for row in tsv_reader:
        if len(row) > 1:
            sentences.append(row[2])

# Step 2: Append the new sentences to the existing file
with open(existing_file_path, mode="a", encoding="utf-8") as file:
    for sentence in sentences:
        file.write(sentence + "\n")
