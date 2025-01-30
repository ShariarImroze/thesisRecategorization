
This code performs a comprehensive topic modeling analysis on a dataset, including data preprocessing, model training, visualization, and various analyses of the resulting topics. It uses advanced NLP techniques like sentence embeddings and BERTopic to uncover latent themes in the text data.

1. Import statements:

   ```python
   import snowflake.connector
   from sqlalchemy import create_engine
   from sqlalchemy.dialects import registry
   import re
   ```
   These lines import necessary libraries for database connections and regular expressions.
2. Credentials setup:

   ```python
   credentials = {
       'account'    : '.azure'
       , 'user'     : 'dgedfdsa'
       , 'authenticator' : 'edfswser'
       , 'role' : 'DEsfdfsdfCS_INGESTOR'
       , 'warehouse' : 'DEfsdfdsE_ANALYTICS_'
   }
   ```
   This dictionary stores credentials for connecting to a Snowflake database.
3. Data import:

   ```python
   import pandas as pd
   df_import = pd.read_sql('''
   select * 
   from DEsdfdsfYTICS.SIfddssRW_Afdsfsdffds_
   ''', snowflake.connector.connect(**credentials))
   ```
   This code imports data from a Snowflake database into a pandas DataFrame.
4. Data preprocessing:

   ```python
   df = df_import.drop_duplicates(subset='CASE_DESCRIPTION').reset_index()
   ```
   This line removes duplicate rows based on the 'CASE_DESCRIPTION' column and resets the index.
5. Document preparation:

   ```python
   docs = df.TITLE_EN.astype(str) + '. ' + list(df.CASE_DESCRIPTION_EN)
   ```
   This creates a list of documents by combining the 'TITLE_EN' and 'CASE_DESCRIPTION_EN' columns.
6. Text cleaning:

   ```python
   docs = [re.sub('Oskarshamn \d - [A-Z0-9]+ - ','', case) for case in docs]
   docs = [re.sub('Oskarshamn \d - ','', case) for case in docs]
   docs = [re.sub('Oskarshamn \d','', case) for case in docs]
   docs = [re.sub('Risk less:','', case) for case in docs]
   ```
   These lines use regular expressions to remove specific patterns from the documents.
7. Sentence embedding:

   ```python
   from sentence_transformers import SentenceTransformer
   sentence_model = SentenceTransformer("all-mpnet-base-v2", device = 'cuda')
   embeddings = sentence_model.encode(docs, show_progress_bar=True)
   ```
   This code uses a pre-trained SentenceTransformer model to create embeddings for each document.
8. NLP setup:

   ```python
   import nltk
   nltk.download('wordnet')
   ```
   This downloads the WordNet lexical database for natural language processing tasks.
9. BERTopic model configuration:

   ```python
   from bertopic import BERTopic
   from bertopic.representation import MaximalMarginalRelevance
   from sklearn.feature_extraction.text import CountVectorizer
   from nltk import word_tokenize    
   from nltk.stem import WordNetLemmatizer 

   class LemmaTokenizer:
       def __init__(self):
           self.wnl = WordNetLemmatizer()
       def __call__(self, doc):
           return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

   vectorizer_model = CountVectorizer(tokenizer=LemmaTokenizer())
   representation_model = KeyBERTInspired()

   topic_model = BERTopic(
       min_topic_size= 7,
       vectorizer_model=vectorizer_model,
       calculate_probabilities=True,
       embedding_model=sentence_model,
       representation_model=representation_model).fit(docs, embeddings)
   ```
   This section sets up the BERTopic model with custom tokenization, vectorization, and representation models.
10. Topic modeling:

    ```python
    topics, probs = topic_model.fit_transform(docs, embeddings)
    ```
    This line fits the topic model to the documents and their embeddings, returning topics and probabilities.
11. Outlier reduction:

    ```python
    topic_reduced = topic_model.reduce_outliers(docs, topics, probabilities=probs, strategy="probabilities", threshold=0.01)
    topics = topic_reduced
    topic_model.update_topics(docs, topics=topic_reduced)
    ```
    These lines reduce outliers in the topic assignments and update the model.
12. Visualization:

    ```python
    topic_model.visualize_topics()
    topic_model.visualize_documents(docs)
    ```
    These lines create visualizations of the topics and documents.
13. Hierarchical topic modeling:

    ```python
    hierarchical_topics = topic_model.hierarchical_topics(docs)
    topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
    ```
    This creates and visualizes a hierarchical structure of the topics.
14. Additional visualizations:

    ```python
    topic_model.visualize_barchart(top_n_topics=8, n_words=8)
    topic_model.visualize_heatmap()
    ```
    These lines create bar charts and heatmaps to visualize the topics.
15. Time-based analysis:

    ```python
    timestamps = df.CASE_OCCURENCE_DATE
    topics_over_time = topic_model.topics_over_time(docs=docs, timestamps=timestamps)
    topic_model.visualize_topics_over_time(topics_over_time, topics=list(range(0,20)))
    ```
    This section analyzes how topics change over time based on the timestamps.
16. Class-based analysis:

    ```python
    classes = df.FUNCTION
    topics_per_class = topic_model.topics_per_class(docs, classes=classes)
    topic_model.visualize_topics_per_class(topics_per_class)
    ```
    This part analyzes how topics are distributed across different classes or categories.
17. Topic distribution analysis:

    ```python
    topic_distr, _ = topic_model.approximate_distribution(docs, min_similarity=0)
    topic_model.visualize_distribution(probs[5])
    topic_model.visualize_distribution(topic_distr[0])
    ```
    These lines calculate and visualize the distribution of topics across documents.
18. Token-level distribution:

    ```python
    topic_distr, topic_token_distr = topic_model.approximate_distribution(docs, calculate_tokens=True)
    df_tl = topic_model.visualize_approximate_distribution(docs[1], topic_token_distr[1])
    ```
    This final section calculates and visualizes the topic distribution at the token level for a specific document.
