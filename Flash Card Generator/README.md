**FLASHCARD GENERATION SYSTEM**

**Flashcard Generator using NLP models:**  
This project incorporates an automated system for Flashcard Generation based on NLP advanced models. It takes input text and returns the question-answer pairs (flashcards that can be used as learning materials).

**The system leverages:**  
- Question Answering (QA) in order to extract an answer from the text.  
- From formulating relevant questions, there is Question Generation (QG).  
- Selective use of Sentence Embedding for collecting quality-based questions.

---

**How It Works**  
1. **Text Tokenization**:  
   Before the processing, the input text is divided into sentences using the NLTK sentence tokenizer.

2. **Question Generation**:  
   To procure questions from text, every sentence is put through the T5 Question Generation model.

3. **Question Selection**:  
   To avoid redundancy, questions are ranked using the similarity score of the sentence embeddings of the questions.

4. **Answer Extraction**:  
   The obtained questions are used for the Roberta-based Question Answering model that is used to retrieve answers from the source text.

5. **Flashcard Creation**:  
   The best question-answer pairs are presented in the form of flashcards for easy revision during the preparation period.

---

**Models Used**  
It leverages advanced NLP models to perform the following tasks:  
- **deepset/roberta-base-squad2**: Applied to Question Answering for extracting the correct answer from the given input text.  
- **valhalla/t5-base-qg-hl**: Employed for Question Generation — an application that generates questions from the text content.  
- **sentence-transformers/paraphrase-MiniLM-L6-v2**: Used for employment and similarity ranking to select diverse and high-quality questions for embedding.

---

**Features**  
- **Automated Question Generation**: Transform input text to potential questions.  
- **Answer Extraction**: Utilizes QA model which provides precise answers from the given text input.  
- **Smart Question Selection**: Recalculates scores of the questions for the similarity with the primary set of questions.  
- **User Input Flexibility**: Enables users to indicate how many flashcards the tool is to create.

---

**Requirements**  
- **nltk**: Text polarities are determined, text pre-processing is done by using Natural Language Toolkit (NLTK) and sentence tokenization implemented.  
- **transformers**: Both the transformers library need to be imported to load the RoBERTa model for Question Answering and T5 model for Question Generation.  
- **torch**: PyTorch is the selected deep learning framework for model inference and processing.  
- **sentence-transformers**: The SentenceTransformer library is used to embed questions and then obtain similarity scores for ranking.

---

**Running the Code**  
1. Input your text when prompted.  
2. Input the number of flashcards that are required to be generated.

**Example**:  

---

**Troubleshooting**  
1. **"Unable to generate any questions"**:  
   Make sure the content of all the input text is meaningful for translation (minimum 1-2 sentences should be included).  

2. **High Memory Usage**:  
   To manage large inputs, please run a script on the system with GPU or decrease text length.

---

**Future Improvements**  
Other enhancements may be as follows:  
- Multilingual option could be provided aiming at expanding the usage base.  
- Feedback from users should also be integrated to make adjustments after recognizing the poor quality of some of the flashcards.  
- Thematic categorization so that the cards may be organized based on the theme of the study.  
- Options in which the length of the flashcards could also be changed.  
- Features like quizzes or timed reviews could be included to make learning lively.  
- Summarization strategies could be applied to extract the summary from the text so that the flashcards to be generated do not overwhelm the learner with information overload.

---

**Purpose and Journey**  
To make studying and revising easier by automating the creation of flashcards. To bring this idea to life, we dove into advanced Natural Language Processing (NLP) models, integrating powerful tools like RoBERTa for answering questions, T5 for generating questions, and Sentence Transformers for embedding and similarity scoring.

The project demanded a lot of experimentation with libraries such as Hugging Face Transformers and PyTorch. A major focus was ensuring smooth communication between multiple models while maintaining a balance between performance and user-friendliness.

---

**Key Challenges and Solutions**  
1. **Handling Model Errors**:  
   Some sentences failed during the question generation process. To address this, I implemented error-handling mechanisms that skipped problematic sentences without interrupting the workflow.

2. **Optimizing Resources**:  
   Running several pre-trained NLP models simultaneously came with significant demands on memory and processing power, especially when dealing with large text inputs. Careful optimization helped keep resource usage in check.

3. **Improving Question Quality**:  
   Ensuring the generated questions were meaningful was no small task. I used sentence embeddings to filter and rank questions, selecting only the most relevant ones for flashcards.

4. **Enhancing Answer Accuracy**:  
   Complex or ambiguous sentences posed challenges for the Question Answering model. Fine-tuning the pipeline significantly improved the accuracy of the extracted answers.

5. **Streamlining Model Management**:  
   Reloading models repeatedly slowed down the process. By saving and reusing models locally, I was able to enhance efficiency and reduce redundancy.

6. **User Interaction and Flexibility**:  
   To make the tool more intuitive, I added input validation and allowed users to set the number of flashcards generated, ensuring the system worked seamlessly within defined limits.

---

**FLASHCARD GENERATION SYSTEM**

--- 
