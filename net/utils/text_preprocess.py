import nltk
import re
from nltk.stem import PorterStemmer
from textblob import Word


stop_words = ['a', 'an', 'the']
typo_corrections = {
    'someth': 'something',
    'selfi': 'selfie',
    'handshak': 'handshake',
    'headach': 'headache',
    'stomachach': 'stomachache',
    'backach': 'backache',
    'neckach': 'neckache',
    'sneez': 'sneeze',
    'salut': 'salute',
    'togeth': 'together',
    # 可以添加更多错别字
}
def stem_word(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word)

def text_preprocess(text):
    # 小写化
    text = text.lower()
    # '/' -> ' or '
    text = text.replace('/', ' or ')
    # 去掉标点和多余空格
    text = re.sub(r'\W+', ' ', text).strip()
    # text = re.sub(r'[^\w\s()]+', ' ', text).strip()
    # 去掉 'a', 'an', 'the'
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # 分词
    tokens = nltk.word_tokenize(text)
    # 词形还原
    # tokens = [stem_word(token) for token in tokens]
    tokens = [Word(token).lemmatize("v") for token in tokens]
    # tokens = lemmatize_with_textblob(tokens)
    # 纠正错别字
    # tokens = [typo_corrections.get(token, token) for token in tokens]
    # return ' '.join(tokens).replace("( ", "(").replace(" )", ")")
    return ' '.join(tokens)

if __name__ == '__main__':
    # 示例
    texts = ["standing up", "stand up", "pick up", "pickup", "typing on a keyboard", "type on keyboard",
             "take off hat or cap", "take off hat/cap", "taking a selfie", "touch head (headache)", "handshaking",
             "sneezing"]
    print(texts)
    processed_texts = [text_preprocess(text) for text in texts]
    print(processed_texts)
