# -*- coding: utf-8 -*-
"""
Created on MAR 01 2019
This script helps count words in sentences labeled "GCS"
@author: Ruiyu Zhang
@version: 20190301
"""
from data_preprocessing import data
import csv

cirno = data(path='D:/CNMC/hospital_data');

cirno.auto_process(merge_unclear=True)
cirno.label_mode = 'l'

datamap = cirno.data
statistics = {}

def includesGCS(l):
    # find if list of labels includes GCS
    for label in l:
        if label.lower().find('gcs') >= 0:
            #print(label)
            return True
    return False

for v in datamap.values():
    if includesGCS(v[3]):
        for word in v[0]:
            statistics[word.lower()] = statistics.get(word.lower(), 0) + 1

with open("D:/CNMC/hospital_data/analyze/GCS_statistics.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["word","count"])
    for k, v in sorted(statistics.items(), key=lambda item: item[1], reverse=True):
        writer.writerow([k, str(v)])

print('>>>Compeleted statistics for GCS')