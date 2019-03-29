import csv
 
with open('whales_predata.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['fname', 'label', 'Sampling (frame) Rate', 'Total Samples (frames)', 'Duration'])
