from pyspark import SparkConf, SparkContext
import json

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
stop_dropped = count_map.filter(lambda a: a[0] not in stopwords).takeOrdered(40, key = lambda a: -a[1])


with open("sp2.json", 'w+') as file_b:
    json.dump(dict(stop_dropped), file_b)

#C
words_stripped = words.filter(lambda a: a.strip(".,:;'!?"))
ws_map = words_stripped.map(lambda word: (word, 1))
ws_counts = ws_map.reduceByKey(lambda a,b: a+b)
ws_count_map = ws_counts.filter(lambda a: a[1]>2)
ws_counts_sorted = ws_count_map.filter(lambda a: a[0] not in stopwords).takeOrdered(40, key = lambda a: -a[1])

with open("sp3.json", 'w+') as file_c:
    json.dump(dict(ws_counts_sorted), file_c)

#D

lines1 = sc.textFile("./data/corpus/4300-0.txt")
lines2 = sc.textFile("./data/corpus/pg36.txt")
lines3 = sc.textFile("./data/corpus/pg514.txt")
lines4 = sc.textFile("./data/corpus/pg1497.txt")
lines5 = sc.textFile("./data/corpus/pg3207.txt")
lines6 = sc.textFile("./data/corpus/pg6130.txt")
lines7 = sc.textFile("./data/corpus/pg19033.txt")
lines8 = sc.textFile("./data/corpus/pg42671.txt")

words1 = lines1.flatMap(lambda line: line.split(" "))
words2 = lines2.flatMap(lambda line: line.split(" "))
words3 = lines3.flatMap(lambda line: line.split(" "))
words4 = lines4.flatMap(lambda line: line.split(" "))
words5 = lines5.flatMap(lambda line: line.split(" "))
words6 = lines6.flatMap(lambda line: line.split(" "))
words7 = lines7.flatMap(lambda line: line.split(" "))
words8 = lines8.flatMap(lambda line: line.split(" "))

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

ws_counts_1 = ws_map_1.reduceByKey(lambda a,b: a+b)
ws_counts_2 = ws_map_2.reduceByKey(lambda a,b: a+b)
ws_counts_3 = ws_map_3.reduceByKey(lambda a,b: a+b)
ws_counts_4 = ws_map_4.reduceByKey(lambda a,b: a+b)
ws_counts_5 = ws_map_5.reduceByKey(lambda a,b: a+b)
ws_counts_6 = ws_map_6.reduceByKey(lambda a,b: a+b)
ws_counts_7 = ws_map_7.reduceByKey(lambda a,b: a+b)
ws_counts_8 = ws_map_8.reduceByKey(lambda a,b: a+b)

ws_count_b_1 = ws_counts_1.filter(lambda a: a[1]>2)
ws_count_b_2 = ws_counts_2.filter(lambda a: a[1]>2)
ws_count_b_3 = ws_counts_3.filter(lambda a: a[1]>2)
ws_count_b_4 = ws_counts_4.filter(lambda a: a[1]>2)
ws_count_b_5 = ws_counts_5.filter(lambda a: a[1]>2)
ws_count_b_6 = ws_counts_6.filter(lambda a: a[1]>2)
ws_count_b_7 = ws_counts_7.filter(lambda a: a[1]>2)
ws_count_b_8 = ws_counts_8.filter(lambda a: a[1]>2)
