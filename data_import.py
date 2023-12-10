from datasets import load_dataset
import csv

dataset = load_dataset("wikipedia", "20220301.simple")
print(type(dataset))
# dataset.save_to_disk("result.json")

dicti = {}

document_count = 1000
count = 0
for row in dataset["train"]:
    dicti[row["id"]]=row["text"]
    count += 1
    # print(row["id"])
    if count==document_count:
        break


import json
with open('ds_dump.json', 'w') as fp:
    json.dump(dicti, fp)

