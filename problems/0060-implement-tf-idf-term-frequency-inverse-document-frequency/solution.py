import numpy as np
import math
from collections import Counter

def compute_tf_idf(corpus, query):
	"""
	Compute TF-IDF scores for a query against a corpus of documents.
    
	:param corpus: List of documents, where each document is a list of words
	:param query: List of words in the query
	:return: List of lists containing TF-IDF scores for the query words in each document
	"""
	
	if not corpus:
		return []

	N = len(corpus)

	df = {word: sum(1 for doc in corpus if word in doc) for word in query}

	idf = {word: math.log((N+1) / (df[word] + 1))+ 1 for word in query}

	tf_idf_scores = []

	for doc in corpus:
		doc_len = len(doc)
		term_counts = Counter(doc)

		doc_scores = []

		for word in query:
			tf = term_counts[word] / doc_len if doc_len > 0 else 0
			tf_idf = tf * idf[word]
			doc_scores.append(round(tf_idf, 5))

		tf_idf_scores.append(doc_scores)

	return tf_idf_scores
