from pyspark import SparkConf, SparkContext
import json
import re
import math

conf = SparkConf().setMaster("local").setAppName("p0")
sc = SparkContext( conf = conf)

lines = sc.textFile("./data/corpus/*")
words_orig = lines.flatMap(lambda line: line.split(" "))
words = words_orig.map(lambda s: s.lower())
word_map = words.map(lambda word: (word, 1))
counts = word_map.reduceByKey(lambda a,b: a+b)

#A
count_map = counts.filter(lambda a: a[1]>2)
counts_sorted = count_map.takeOrdered(40, key = lambda a: -a[1])

with open("sp1.json", 'w+') as file_a:
    json.dump(dict(counts_sorted), file_a)

#B
stopwords = sc.textFile("./data/stopwords.txt").collect()
sw = sc.broadcast(stopwords)
stop_dropped = count_map.filter(lambda a: a[0] not in sw.value).takeOrdered(40, key = lambda a: -a[1])


with open("sp2.json", 'w+') as file_b:
    json.dump(dict(stop_dropped), file_b)

#C
words_stripped = words.filter(lambda a: a.strip(".,:;'!?"))
ws_map = words_stripped.map(lambda word: (word, 1))
ws_counts = ws_map.reduceByKey(lambda a,b: a+b)
ws_count_map = ws_counts.filter(lambda a: a[1]>2)
ws_counts_sorted = ws_count_map.filter(lambda a: a[0] not in sw.value).takeOrdered(40, key = lambda a: -a[1])

with open("sp3.json", 'w+') as file_c:
    json.dump(dict(ws_counts_sorted), file_c)

#D
#
lines = sc.wholeTextFiles("./data/corpus/*")

lines1 = sc.textFile("./data/corpus/4300-0.txt")
lines2 = sc.textFile("./data/corpus/pg36.txt")
lines3 = sc.textFile("./data/corpus/pg514.txt")
lines4 = sc.textFile("./data/corpus/pg1497.txt")
lines5 = sc.textFile("./data/corpus/pg3207.txt")
lines6 = sc.textFile("./data/corpus/pg6130.txt")
lines7 = sc.textFile("./data/corpus/pg19033.txt")
lines8 = sc.textFile("./data/corpus/pg42671.txt")

words_temp = lines.map(lambda a:(a[0], filter(None, a[1].encode('ascii', errors='ignore')
                                    .strip()
                                    .decode('ascii')
                                    .replace("\r", " ")
                                    .replace("\n", " ")
                                    .lower()
                                    .split(" "))))

words1 = lines1.flatMap(lambda line: line.lower().split(" "))
words2 = lines2.flatMap(lambda line: line.lower().split(" "))
words3 = lines3.flatMap(lambda line: line.lower().split(" "))
words4 = lines4.flatMap(lambda line: line.lower().split(" "))
words5 = lines5.flatMap(lambda line: line.lower().split(" "))
words6 = lines6.flatMap(lambda line: line.lower().split(" "))
words7 = lines7.flatMap(lambda line: line.lower().split(" "))
words8 = lines8.flatMap(lambda line: line.lower().split(" "))


words1_stripped = words1.filter(lambda a: a.strip(".,:;'!?"))
words2_stripped = words2.filter(lambda a: a.strip(".,:;'!?"))
words3_stripped = words3.filter(lambda a: a.strip(".,:;'!?"))
words4_stripped = words4.filter(lambda a: a.strip(".,:;'!?"))
words5_stripped = words5.filter(lambda a: a.strip(".,:;'!?"))
words6_stripped = words6.filter(lambda a: a.strip(".,:;'!?"))
words7_stripped = words7.filter(lambda a: a.strip(".,:;'!?"))
words8_stripped = words8.filter(lambda a: a.strip(".,:;'!?"))


ws_map_1 = words1_stripped.map(lambda word: (word, 1))
ws_map_2 = words2_stripped.map(lambda word: (word, 1))
ws_map_3 = words3_stripped.map(lambda word: (word, 1))
ws_map_4 = words4_stripped.map(lambda word: (word, 1))
ws_map_5 = words5_stripped.map(lambda word: (word, 1))
ws_map_6 = words6_stripped.map(lambda word: (word, 1))
ws_map_7 = words7_stripped.map(lambda word: (word, 1))
ws_map_8 = words8_stripped.map(lambda word: (word, 1))
#
ws_counts_1 = ws_map_1.reduceByKey(lambda a,b: a+b).filter(lambda b: b[1]>1)
ws_counts_2 = ws_map_2.reduceByKey(lambda a,b: a+b).filter(lambda b: b[1]>1)
ws_counts_3 = ws_map_3.reduceByKey(lambda a,b: a+b).filter(lambda b: b[1]>1)
ws_counts_4 = ws_map_4.reduceByKey(lambda a,b: a+b).filter(lambda b: b[1]>1)
ws_counts_5 = ws_map_5.reduceByKey(lambda a,b: a+b).filter(lambda b: b[1]>1)
ws_counts_6 = ws_map_6.reduceByKey(lambda a,b: a+b).filter(lambda b: b[1]>1)
ws_counts_7 = ws_map_7.reduceByKey(lambda a,b: a+b).filter(lambda b: b[1]>1)
ws_counts_8 = ws_map_8.reduceByKey(lambda a,b: a+b).filter(lambda b: b[1]>1)

words_stripped_temp = words_temp.map(lambda a: (a[0], [x.strip(".,:;'!? ") for x in a[1]]))
final_words = words_stripped_temp.map(lambda a: (a[0], list(set(a[1]))))

doc_freq = final_words.flatMap(lambda a: a[1]).map(lambda b: (b, 1)).reduceByKey(lambda a,b: a+b)

tfdf_1 = ws_counts_1.join(doc_freq)
tfdf_2 = ws_counts_2.join(doc_freq)
tfdf_3 = ws_counts_3.join(doc_freq)
tfdf_4 = ws_counts_4.join(doc_freq)
tfdf_5 = ws_counts_5.join(doc_freq)
tfdf_6 = ws_counts_6.join(doc_freq)
tfdf_7 = ws_counts_7.join(doc_freq)
tfdf_8 = ws_counts_8.join(doc_freq)

tfidf_1 = tfdf_1.map(lambda a: (a[0], a[1][0] * math.log(8 / a[1][1]))).takeOrdered(5, key = lambda b: -b[1])
tfidf_2 = tfdf_2.map(lambda a: (a[0], a[1][0] * math.log(8 / a[1][1]))).takeOrdered(5, key = lambda b: -b[1])
tfidf_3 = tfdf_3.map(lambda a: (a[0], a[1][0] * math.log(8 / a[1][1]))).takeOrdered(5, key = lambda b: -b[1])
tfidf_4 = tfdf_4.map(lambda a: (a[0], a[1][0] * math.log(8 / a[1][1]))).takeOrdered(5, key = lambda b: -b[1])
tfidf_5 = tfdf_5.map(lambda a: (a[0], a[1][0] * math.log(8 / a[1][1]))).takeOrdered(5, key = lambda b: -b[1])
tfidf_6 = tfdf_6.map(lambda a: (a[0], a[1][0] * math.log(8 / a[1][1]))).takeOrdered(5, key = lambda b: -b[1])
tfidf_7 = tfdf_7.map(lambda a: (a[0], a[1][0] * math.log(8 / a[1][1]))).takeOrdered(5, key = lambda b: -b[1])
tfidf_8 = tfdf_8.map(lambda a: (a[0], a[1][0] * math.log(8 / a[1][1]))).takeOrdered(5, key = lambda b: -b[1])


with open("sp4.json", "w+") as file_d:
    json.dump(dict(tfidf_1 + tfidf_2 + tfidf_3 + tfidf_4 + tfidf_5 + tfidf_6 + tfidf_7 + tfidf_8), file_d)
