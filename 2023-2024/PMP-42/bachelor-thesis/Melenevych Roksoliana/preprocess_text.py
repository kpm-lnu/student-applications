from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
	tokens = word_tokenize(text)
	tokens = [word for word in tokens if word.isalpha()]
	tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
	lemmatizer = WordNetLemmatizer()
	tokens = [lemmatizer.lemmatize(word) for word in tokens]
	return ' '.join(tokens)