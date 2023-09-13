import string
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

class key_extractor():
    def __init__(self):
       nltk.download('stopwords')
       self.stop_words = stopwords.words('english')

    def get_key(self,text):
        # 去除标点和数字
        text = text.translate(str.maketrans('', '', string.punctuation + string.digits))

        # 将文本转换成小写
        text = text.lower()

        # 去除停用词
        words = text.split()
        words = [word for word in words if word not in self.stop_words]

        # 计算词频
        word_counts = Counter(words)

        # 计算TF-IDF值
        vectorizer = TfidfVectorizer(max_features=10)
        tfidf = vectorizer.fit_transform(words)
        tfidf_scores = zip(vectorizer.get_feature_names_out(), tfidf.toarray()[0])
        tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)

        # 提取关键词
        keywords = [word[0] for word in tfidf_scores]

        # print(keywords)

        return keywords
    
# job_requirement = "Diverse role with close cooperation with the Managing Director and team.Fantastic team culture.Have oversight over a diverse range of accounts."
# key_etc = key_extractor()
# job_key = key_etc.get_key(job_requirement)
# print(job_key)