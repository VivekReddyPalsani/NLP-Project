# **Keyphrase Extraction Model**

## **Introduction**  
Keyphrase extraction is a critical task in natural language processing, especially for academic and scientific texts, where identifying key concepts can significantly enhance learning and comprehension. The development of a **fine-tuned SciBERT-based keyphrase extraction model** aims to provide students with a tool to efficiently extract key ideas from lengthy academic content. 

This model complements a suite of NLP tools designed to assist students in academic learning, such as summarization, question generation, and content indexing. By automating the extraction of keyphrases, this tool empowers students to focus on critical concepts, enhancing their overall learning experience.

---

## **Development Process**  

### **1. Initial Exploration**  
The journey began with a Seq2Seq model for keyphrase extraction. While this approach showed promise, challenges such as irrelevant outputs, vocabulary management issues, and limited contextual accuracy prompted a transition to alternative methods like KeyBERT. KeyBERT, based on static embeddings, provided better results but lacked adaptability to academic tasks.

### **2. Final Approach: SciBERT**  
To address these challenges, the final model leveraged **SciBERT**, a domain-specific transformer model pretrained on scientific text. SciBERT was fine-tuned for token classification using the **SemEval-2010 dataset**, which includes BIO-tagged data (`B`, `I`, `O`) for keyphrase extraction. This approach significantly improved contextual accuracy and relevance.

---

## **Evaluation and Results**  

### **Example 1**  
- **Input**: "Deep learning is a key component of modern artificial intelligence."  
- **Extracted Keyphrases**: ["deep learning", "artificial intelligence", "key component"]  

### **Example 2**  
- **Input**: "It's a software program that manages all other application programs in a computer. The OS acts as a bridge between hardware and software, and allows users to interact with the computer."  
- **Extracted Keyphrases**: [“software program”, “application programs”, “os”, “hardware”, “software”, “users”, “computer”] 

### **Analysis of Results**  
- **Contextual Accuracy**: Captures domain-specific vocabulary (e.g., "photosynthesis").  
- **Precision in Keyphrases**: Removes irrelevant or overly generic terms.  
- **Fluency in Extraction**: Combines fragmented subwords into meaningful phrases (e.g., "deep learning").  

---

## **Collaborative Models and Use Cases**  
This model complements other NLP tools to enhance academic learning:  
1. **Summarization**: Condenses large texts for easier review.  
2. **Question Generation**: Creates questions for self-assessment and practice.  
3. **Content Indexing**: Organizes academic resources for
