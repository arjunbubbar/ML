import pandas as pd

text = pd.read_csv ('/Users/arjunbubbar/Desktop/Jetlearn/Data Science/Datasets/sentiments.txt', sep=';',names = ['sentence','sentiment'])

print (text.info ())
print (text ['sentiment'].value_counts ())

text ['sentiment'] = text ['sentiment'].replace ({'joy':1,'sadness':0,'anger':0,'fear':0,'love':1,'surprise':1})

print (text ['sentiment'].value_counts ())