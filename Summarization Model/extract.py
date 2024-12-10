!pip install nltk
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
nltk.download("punkt_tab")
nltk.download("stopwords")
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9\s]', '', text))
    return text
def tokenize_sentences(text):
    return sent_tokenize(text)
def calculate_sentence_scores(sentences, text):
    stop_words = set(stopwords.words("english"))
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)

    sentence_scores = tfidf_matrix.sum(axis=1).A1
    sentence_score_dict = {i: sentence_scores[i] for i in range(len(sentences))}
    return sentence_score_dict
def generate_summary(text, num_sentences=3):
    sentences = tokenize_sentences(text)
    sentence_scores = calculate_sentence_scores(sentences, text)

    ranked_sentences = sorted(sentence_scores.keys(), key=lambda x: sentence_scores[x], reverse=True)

    top_sentences = [sentences[i] for i in ranked_sentences[:num_sentences]]
    summary = " ".join(top_sentences)
    return summary
if __name__ == "__main__":
    # Input text
    input_text = """
    Text summarization is usually implemented by natural language processing methods,
    designed to locate the most informative sentences in a given document.
    On the other hand, visual content can be summarized using computer vision algorithms.
    Image summarization is the subject of ongoing research;
    existing approaches typically attempt to display the most representative images from a given image collection,
    or generate a video that only includes the most important content from the entire collection.
    Video summarization algorithms identify and extract from the original video content the most important frames (key-frames),
    and/or the most important video segments (key-shots),
    normally in a temporally ordered fashion.
    Video summaries simply retain a carefully selected subset of the original video frames and,
    therefore, are not identical to the output of video synopsis algorithms,
    where new video frames are being synthesized based on the original video content.
    """

    # Generate the summary
    summary = generate_summary(input_text, num_sentences=2)
    print("Original Text:\n", input_text)
    print("\nGenerated Summary:\n", summary)
