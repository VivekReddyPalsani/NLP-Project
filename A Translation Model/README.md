### Academic Text Translation Model

---

**Introduction**  
Language is often a significant obstacle for students when accessing academic resources, especially in multilingual countries like India. The need for inclusivity in education inspired the development of a fine-tuned **English-to-Telugu translation model** using the mBART framework. This project bridges language gaps by enabling Telugu-speaking students to access academic content in their native language.

Our translation model is part of a larger suite of NLP tools developed to enhance accessibility to academic materials. These tools include key phrase extraction, summarization, question generation, and a word assistance tool for extracting and simplifying difficult words.

---

### **Development Process**  

**1. Dataset and Challenges**  
We initially aimed to use datasets like Wiki40B and scraped Wikipedia articles for English-Telugu translation pairs. However, these datasets proved unsuitable for several reasons:  
- **Inconsistent Translations**: Wikipedia often paraphrases text, reducing reliability.  
- **Structural Variability**: Differences in layout complicated alignment.  
- **Encoding Issues**: Variations in script encoding created tokenization challenges.  

To address these issues, we adopted the **Telugu-LLM-Labs/Telugu_Teknium_GPTeacher** dataset, which contains curated, aligned English-Telugu sentence pairs. This dataset offered domain-specific relevance and alignment crucial for fine-tuning.

---

### **Evaluation and Results**  

1. **Example 1**:  
   **English**: "The gravitational force between two objects is directly proportional to their masses."  
   - **Baseline Translation**: "రెండు వస్తువుల మధ్య Gravitational శక్తి వారి మాంసలు నేరుగా సమతుల్య ఉంది."  
   - **Fine-Tuned Translation**: "రెండు వస్తువుల మధ్య కర్షణ శక్తి వారి సమూహాలకు నేరుగా సారస్యంగా ఉంటుంది."  
   - **Improvement**: The fine-tuned model accurately translated "gravitational force" to "కర్షణ శక్తి" and corrected "masses" to "సమూహాలకు" (correct scientific term) instead of the incorrect "మాంసలు" (meat) used by the baseline model.  

2. **Example 2**:  
   **English**: "Photosynthesis is the process by which green plants convert sunlight into chemical energy."  
   - **Baseline Translation**: "Photosynthesis ద్వారా ఆకుపచ్చ మొక్కలు, ரசాయన శక్తి కు సూర్యకాంతి కవదిలే процెసి."  
   - **Fine-Tuned Translation**: "కిరణజన్య సంయోగ అనేది ఆకుపచ్చ మొక్కలు సూర్యరశ్మిని రసామి శక్తిగా మార్చే ప్రక్రియ."  
   - **Improvement**: The fine-tuned model translated "photosynthesis" to the correct Telugu term "కిరణజన్య సంయోగ" and eliminated script inconsistencies where Tamil and Telugu characters were mixed.  

3. **Example 3**:  
   **English**: "Newton's third law states that every action has an equal and opposite reaction."  
   - **Baseline Translation**: "న్యూటన్ యొక్క మూడవ చట్టం ప్రతిచర్య ఒక సగటు మరియు విరుద్ధ చర్య కలిగి states."  
   - **Fine-Tuned Translation**: "న్యూటన్ యొక్క మూడవ చట్టం ప్రతి చర్య సమానమైన మరియు విరుద్ధ ప్రతిచర్యను కలిగి ఉంటుంది."  
   - **Improvement**: The fine-tuned model handled the phrase "equal and opposite reaction" with better contextual accuracy and eliminated untranslated words like "states" present in the baseline translation.  

**Analysis of Improvements**  
- **Domain-Specific Vocabulary**: Accurate translations of technical terms such as "gravitational force" and "photosynthesis."  
- **Contextual Precision**: Retained the academic tone and ensured the content's scientific integrity.  
- **Language Consistency**: Eliminated mixed scripts and untranslated words.  
- **Grammatical Fluency**: Produced coherent, grammatically correct translations, maintaining meaning and readability.

---

### **Collaborative Models and Use Cases**  
Our translation model complements a suite of other NLP tools designed to support academic learning:  
1. **Key Phrase Extraction**: Highlights critical concepts for focused learning.  
2. **Summarization**: Condenses long passages for quick review.  
3. **Question Generation**: Creates questions from content for self-assessment.  
4. **Word Assistance Tool**: Extracts difficult terms, provides definitions, and suggests synonyms.  

These tools collectively empower students by offering a multilingual, accessible approach to education.  

---

### **Conclusion and Future Scope**  
The fine-tuned mBART model successfully bridges language barriers for Telugu-speaking students, offering accurate and contextually relevant translations for academic content. Future enhancements include:  
- Expanding to other regional languages like Tamil and Hindi.  
- Training on larger datasets using high-performance computing resources.  
- Integrating these tools into digital learning platforms for broader accessibility.

---

### **References**  
1. Hugging Face Transformers Documentation. [https://huggingface.co/docs/transformers/](https://huggingface.co/docs/transformers/)  
2. OPUS100 Dataset. [https://opus.nlpl.eu/](https://opus.nlpl.eu/)  
3. Telugu-LLM-Labs Dataset. [https://huggingface.co/datasets/Telugu-LLM-Labs/telugu_teknium_GPTeacher_general_instruct_filtered_romanized](https://huggingface.co/datasets/Telugu-LLM-Labs/telugu_teknium_GPTeacher_general_instruct_filtered_romanized)  

---

**Note**: Some assistance was taken from ChatGPT to refine code and generate specific components for this project.
