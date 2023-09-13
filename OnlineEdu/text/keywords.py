import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


class KeyExtractor():
    def __init__(self):
       nltk.download('stopwords')
       self.stop_words = stopwords.words('english')

    def get_key(self,text):
        # text preprocess
        text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
        text = text.lower()
        words = text.split()
        words = [word for word in words if word not in self.stop_words]

        # calculate TF-IDF value
        vectorizer = TfidfVectorizer(max_features=10)
        tfidf = vectorizer.fit_transform(words)
        tfidf_scores = zip(vectorizer.get_feature_names_out(), tfidf.toarray()[0])
        tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)

        # extract keywords
        keywords = [word[0] for word in tfidf_scores]
        return keywords
