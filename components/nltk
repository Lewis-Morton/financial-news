import streamlit as st 
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import bigrams, trigrams, FreqDist
import pandas as pd

def render(clean_words, word_series):
    #most common word, bigrams, and trigrams are plotted using seaborn barplots, tokenisation, removal of punctuation and 
    #stopwords was carried out in jupyter notebooks
    
    st.write('To better understand the language patterns within the headlines, we examined the frequency of individual'
                'words as well as common word pairings (bigrams) and triplets (trigrams). This analysis highlights recurring' 
                ' themes and keywords that may reveal the focus or tone of the dataset.')

    #get count of each unique word in word_series, select top 20 words, and 2 new columns for the df
    most_common_words = word_series.value_counts().head(20).reset_index()
    most_common_words.columns = ['Word', 'Count']
        
        
    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(data=most_common_words.head(20), x='Word', y='Count', palette='Set2', edgecolor='black', linewidth=2, ax=ax)
    ax.set_title('Top 20 Most Common Headline Words')
    ax.set_ylabel('Count')
    ax.set_xlabel('Words')
    ax.grid(True, axis='y')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.write(
            'This chart shows the 20 most frequently used words across all headlines. It gives a clear picture of which terms'
            'dominate the dataset, providing insight into the most emphasised topics or framing.'
        )

    #create lists of bigrams and tridgrams from clean_words
    bigrams_list = list(bigrams(clean_words))
    trigrams_list = list(trigrams(clean_words))

    #create bigram and trigram frequency distributions
    fdist_bigrams = FreqDist(bigrams_list)
    fdist_trigrams = FreqDist(trigrams_list)

    #extract 10 most common bigrams and trigrams as lists of tuples
    top10_bigrams = fdist_bigrams.most_common(10)
    top10_trigrams = fdist_trigrams.most_common(10)

    #build dfs for bigrams and trigrams with 2 columns, one for the collocation and one for the count
    df_bigrams = pd.DataFrame(top10_bigrams, columns=['Bigram', 'Count'])
    df_trigrams = pd.DataFrame(top10_trigrams, columns=['Trigram', 'Count'])

    #use the join method to remove the separator from within the tuples of bigrams and trigrams
    df_bigrams['Bigram'] = [' '.join(x) for x in df_bigrams['Bigram']]
    df_trigrams['Trigram'] = [' '.join(x) for x in df_trigrams['Trigram']]

    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(data=df_bigrams.head(10), x='Bigram', y='Count', palette='Set2', edgecolor='black', linewidth=2, ax=ax)
    ax.set_title('Top 10 Most Common Bigrams')
    ax.set_ylabel('Count')
    ax.set_xlabel('Bigrams')
    ax.grid(True, axis='y')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.write(
            'Here we look at the most common two-word combinations found in the headlines. These bigrams often highlight relationships'
            'between concepts or recurring phrases, offering more context than single words alone.'
        )

    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(data=df_trigrams.head(10), x='Trigram', y='Count', palette='Set3', edgecolor='black', linewidth=2, ax=ax)
    ax.set_title('Top 10 Most Common Trigrams')
    ax.set_ylabel('Count')
    ax.set_xlabel('Trigrams')
    ax.grid(True, axis='y')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.write('This chart extends the analysis to three-word sequences. Trigrams help capture headline structures and recurring'
                 'expressions, giving a deeper view into how ideas are framed or repeated.')

    #st.subheader('Top 10 Bigrams')
    #st.dataframe(df_bigrams)

    #st.subheader('Top 10 Trigrams')
    #st.dataframe(df_trigrams)
        
