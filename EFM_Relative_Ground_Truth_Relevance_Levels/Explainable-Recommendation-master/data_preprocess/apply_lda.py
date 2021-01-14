from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from gensim import corpora, models
import gensim
import sys
import os
import json
import re

stops = set(stopwords.words("english"))
p_stemmer = PorterStemmer()

def extract_reviews(file):
    copous = []
    with open(file) as data_file:
        for line in data_file:
            json_data = json.loads(line)
            review_text = json_data["reviewText"]
            review_text = review_text.lower()
            review_text = re.sub(r"[^A-Za-z0-9\'\s]", "", review_text)
            tokens = word_tokenize(review_text)
            stopped_tokens = [i for i in tokens if not i in stops]
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
            copous.append(stemmed_tokens)
    return copous
if len(sys.argv) < 3:
    sys.exit('Usage: %s data-file output-dir' % sys.argv[0])
output_path = sys.argv[2]
data_file = sys.argv[1]
#if not os.path.splitext(file)[1] == '.json':
#    sys.exit('input data should be a json file.')
texts = extract_reviews(data_file)
# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word=dictionary, passes=50)
print(ldamodel.print_topics(num_topics=10, num_words=4))