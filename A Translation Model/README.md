### Academic Text Translation Model

---

**Introduction**  
Language is often a significant obstacle for students when accessing academic resources, especially in multilingual countries like India. The need for inclusivity in education inspired the development of a fine-tuned **English-to-Telugu translation model** using the mBART framework. This project bridges language gaps by enabling Telugu-speaking students to access academic content in their native language.

Our translation model is part of a larger suite of NLP tools developed to enhance accessibility to academic materials. These tools include key phrase extraction, summarization, question generation, and a word assistance tool for extracting and simplifying difficult words.

---

### **Why mBART and Approach Overview**

**Why mBART?**  
The **mBART (Multilingual Bidirectional and Auto-Regressive Transformer)** model was chosen because of its advanced architecture and suitability for multilingual tasks. Key reasons include:  
1. **Multilingual Capability**:
   - mBART is pre-trained on a diverse set of languages, making it ideal for translation tasks involving Indian regional languages like Telugu.  
   - It supports over 50 languages, enabling seamless language transitions.  

2. **Pretraining Objective**:
   - mBART is trained using a **denoising autoencoder objective**, which reconstructs corrupted text sequences. This equips it with robust context understanding and translation accuracy.  

3. **Fine-Tuning Efficiency**:
   - The model can be fine-tuned on specific language pairs (e.g., English-Telugu) using high-quality, domain-specific datasets.  
   - It allows the integration of specialized datasets to adapt the model for academic language nuances.  

---

**Approach Overview**  
The implementation process involved the following key steps:

1. **Dataset Preparation**:
   - The **Telugu-LLM-Labs/Telugu_Teknium_GPTeacher** dataset was used, providing aligned English-Telugu sentence pairs.
   - Preprocessing included tokenization, cleaning, and normalization to ensure compatibility with the mBART tokenizer.

2. **Model Selection**:
   - The **facebook/mbart-large-50-many-to-many-mmt** model was chosen for its ability to handle multilingual tasks effectively.
   - The tokenizer was configured for **English** (`en_XX`) as the source language and **Telugu** (`te_IN`) as the target language.

3. **Fine-Tuning**:
   - **Training Configuration**:
     - Training and validation datasets were prepared with a 90:10 split.
     - Gradient accumulation was used to manage smaller batch sizes due to hardware limitations.
   - **Training Process**:
     - The model was trained on Google Colab using CPU/MPS for efficient computation.
     - Hyperparameters like learning rate (5e-5), batch size (4), and evaluation steps (10) were carefully tuned.
   - **Evaluation**:
     - BLEU and METEOR scores were periodically calculated to monitor translation quality.

4. **Model Deployment**:
   - After fine-tuning, the model was saved for deployment in academic and educational applications.

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

### **Model Download and Usage Instructions**

---

#### **Model Download**
Download the fine-tuned mBART model from the link below and save it to your directory:

**[Download Model](https://mahindraecolecentrale-my.sharepoint.com/:f:/g/personal/se22uari122_mahindrauniversity_edu_in/EtnsNzet-h5KqarxFxHxllUBQW19CPlobYdbh0WMcc72AQ?e=kHuzHE)**

---

#### **How to Use**
1. **Install Dependencies**:
   Run the following command to install required libraries:
   ```bash
   pip install transformers torch
   ```

2. **Run the Script**:
   Save the `translate.py` script and ensure it is in the same directory as the downloaded model. Then, execute:
   ```bash
   python translate.py
   ```

---

#### **Example Usage**
The script includes a test sentence:
```python
english_text = "Newton's third law states that every action has an equal and opposite reaction."
```
Running the script outputs:
```text
Translated Text: న్యూటన్ యొక్క మూడవ చట్టం ప్రతి చర్య సమానమైన మరియు విరుద్ధ ప్రతిచర్యను కలిగి ఉంటుంది.
```

---

#### **To Test Custom Sentences**
Edit the `english_text` variable in the script with your text and rerun:
```python
english_text = "Your English sentence here."
```

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
