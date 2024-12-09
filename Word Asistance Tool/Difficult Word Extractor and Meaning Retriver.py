import pandas as pd
import re
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize

file_path = '/Users/vigneshanoop/Downloads/WordDifficulty.csv'
df = pd.read_csv(file_path)

stop_words = set(stopwords.words("english"))

easy_words = df[(df['I_Mean_Accuracy'] > 0.7) & (df['I_Mean_RT'] < 700)]['Word'].str.lower().tolist()
medium_words = df[((df['I_Mean_Accuracy'] >= 0.5) & (df['I_Mean_Accuracy'] <= 0.7)) |
                  ((df['I_Mean_RT'] >= 700) & (df['I_Mean_RT'] <= 900))]['Word'].str.lower().tolist()
difficult_words = df[(df['I_Mean_Accuracy'] < 0.5) | (df['I_Mean_RT'] > 900)]['Word'].str.lower().tolist()

def get_word_meaning_with_pos(word):
    synsets = wordnet.synsets(word)
    if synsets:
        meanings_with_pos = []
        for syn in synsets[:3]:  
            pos = syn.pos()    
            definition = syn.definition()
            meanings_with_pos.append((pos, definition))
        return meanings_with_pos
    else:
        return [("N/A", "Meaning not found")]

def display_words_from_text(text, difficulty_level):
    tokens = word_tokenize(text.lower())
    words = set(word for word in tokens if word.isalpha() and word not in stop_words)
    
    if difficulty_level == 1:
        selected_words = [word for word in words if word in easy_words]
    elif difficulty_level == 2:
        selected_words = [word for word in words if word in medium_words]
    elif difficulty_level == 3:
        selected_words = [word for word in words if word in difficult_words]
    else:
        print("Invalid difficulty level entered.")
        return
    
    non_listed_words = [word for word in words if word not in easy_words + medium_words + difficult_words]
    
    print(f"\nWords and Meanings for Difficulty Level {difficulty_level}:")
    for word in selected_words + non_listed_words:
        meanings = get_word_meaning_with_pos(word)
        print(f"{word.capitalize()}:")
        for pos, meaning in meanings:
            print(f"  ({pos}) {meaning}")
        print() 
      
text_input = "The enigmatic cryptographer, deciphering the elusive code, inadvertently revealed the long-lost secret, altering the course of history."
difficulty_level = 3  # Use 1 for easy, 2 for medium, and 3 for difficult words
display_words_from_text(text_input, difficulty_level)
