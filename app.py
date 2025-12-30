import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

#load preprocessed datasets and cleaned word lists from jupyter for use in the dashboard
df = pd.read_csv('data/financial_news_events.csv')
df_clean = pd.read_csv('data/financial_news_events_clean.csv')
heatmap_df = pd.read_csv('data/financial_news_events_heatmap_df.csv')
clean_words = pd.read_csv('data/clean_words.csv')['Word'].tolist()
word_series = pd.Series(clean_words)
df_one_hot_encoded = pd.read_csv('data/df_one_hot_encoded.csv')

#without setting the layout to wide, only 6 tabs are rendered
st.set_page_config(layout='wide')

st.title('Financial News Dashboard', width='stretch')
st.divider()
st.header('Analyse the Effects of Financial News on Markets', width='stretch')
st.text('In the following analysis we look at the effects of financial news headlines on markets')
st.text('The aim of the analysis is to uncover market trends which could serve traders who wish factor in the impact of news events on market performance and trading patterns')
#expander keeps key information above the fold
with st.expander('See more details'):
    st.text('Columns in the dataset include: \n-date\n-headline source\n-market event\n-market index\n-index change percent\n-trading volume\n-sector\n-sentiment\n-impact level\n-related company\n-news url')
    

st.sidebar.header('Financial News Controls')

#date range filter
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime(df_clean['Date']).min())
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime(df_clean['Date']).max())

#sentiment filter
selected_sentiments = st.sidebar.multiselect(
    "Sentiment",
    options=df_clean['Sentiment'].unique(),
    default=df_clean['Sentiment'].unique()
)

#sector filter
selected_sectors = st.sidebar.multiselect(
    "Sector",
    options=df_clean['Sector'].unique(),
    default=df_clean['Sector'].unique()
)

#impact level filter
selected_impact = st.sidebar.multiselect(
    'Impact Level',
    options=['Low', 'Medium', 'High'],
    default=['Low', 'Medium', 'High']
)

#create new df to avoid changing original df_clean
df_filtered = df_clean.copy()

#date filter
df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])
df_filtered = df_filtered[(df_filtered['Date'] >= pd.to_datetime(start_date)) &
                          (df_filtered['Date'] <= pd.to_datetime(end_date))]

#sentiment filter
df_filtered = df_filtered[df_filtered['Sentiment'].isin(selected_sentiments)]

#sector filter
df_filtered = df_filtered[df_filtered['Sector'].isin(selected_sectors)]

#impact level filter
if selected_impact:
    df_filtered = df_filtered[df_filtered['Impact_Level'].isin(selected_impact)]

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['Overview', 'Distributions', 'Categorical Analysis', 'Market Movement analysis', 'Correlations', 'Headline Text Analysis', 'Predictive Modelling'])

with tab1:

    col1, col2, col3, col4, col5 = st.columns(5)
    #key metrics at a glance
    col1.metric('Total Headlines: ', len(df_filtered))
    col2.metric('Average Index Change Percent: ', f'{round(df_filtered["Index_Change_Percent"].mean(), 2)}%')
    col3.metric('Most Common Sentiment: ', df_filtered['Sentiment'].mode()[0])
    col4.metric('Most Impacted Sector: ', df_filtered['Sector'].mode()[0])
    col5.metric('Most Common Event Type: ', df_filtered['Market_Event'].mode()[0])

    st.write('Original Dataset Shape: ', df.shape)

    st.write('Missing Values Before Data cleaning:')
    missing_per_column = df.isna().sum().reset_index()
    missing_per_column.columns = ['Column', 'Missing Values']
    st.dataframe(missing_per_column)

    st.write('Rows missing Headline or Index_Change_Percent were removed. Missing sentiment scores were filled using sentiment'
            'analysis on the headlines, and missing URL values were replaced with \'missing url data\'.')
    
    st.write('Final Dataset Shape: ', df_clean.shape)
    
    st.write('Summary Statistics Post Data Cleaning: \n')
    st.write(df_clean.describe())

    st.write('Clean Dataset Preview: ')
    st.dataframe(df_clean.head())

import seaborn as sns

with tab2:
    #contains distributions of key features within the dataset as histograms and bar charts
    st.write('**How are key news and market features distributed over time?**')
    st.write(
        'This section explores how news sentiment, trading activity, market movements, and event impact are distributed within'
        'the dataset. These distributions provide context on what types of events are most common and how market responses are'
        'typically spread.'
    )
    st.write(
        '**Note:** The visualisations in this tab reflect the currently selected date range. Adjusting the date filter may change'
          'the shape of these distributions, highlighting how news and market behaviour vary across different periods.'
    )

    fig, ax = plt.subplots(figsize=(12,6))
    sns.countplot(x='Sentiment', data=df_filtered, palette='Set2', edgecolor='black', linewidth=2, ax=ax)
    ax.set_title('Sentiment Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('News Sentiment')
    ax.grid(True, axis='y')
    st.pyplot(fig)
    
    st.write(
        'This chart shows how news headlines are distributed across positive, neutral, and negative sentiment categories within'
        'the selected time period. A skew toward a particular sentiment may indicate prevailing market narratives during that period.'
    )
   

    fig, ax = plt.subplots(figsize=(12, 6))
    df_filtered['Trading_Volume'].plot(kind='hist', bins=10, ax=ax, color='mistyrose', edgecolor='black', linewidth=2)
    ax.set_title('Trading Volume Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('Trading Volume')
    ax.grid(True, axis='y')
    st.pyplot(fig)
    
    st.write(
        'The histogram illustrates how often different trading volume ranges occur. Concentration around certain values may indicate common'
        'trading levels while long tails could indicate unusual trading activity.'
        
    )


    fig, ax = plt.subplots(figsize=(12,6))
    df_filtered['Index_Change_Percent'].plot(kind='hist', bins=10, ax=ax, color='honeydew', edgecolor='black', linewidth=2)
    ax.set_title('Index Change % Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('Index Change Percent')
    ax.grid(True, axis='y')
    st.pyplot(fig)
    
    st.write(
        'This chart displays the distribution of market returns following news events. The shape of the distribution provides'
        'insight into the frequency and magnitude of positive and negative market movements.'
    )


    fig, ax = plt.subplots(figsize=(12,6))
    sns.countplot(x='Impact_Level', data=df_filtered, palette='Set3', edgecolor='black', linewidth=2, ax=ax)
    ax.set_title('Impact Level Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('Impact Level')
    ax.grid(True, axis='y')
    st.pyplot(fig)                    
    
    st.write(
        'This chart shows how event impact scores are distributed across news events. A concentration of low or high impact values'
        'can suggest whether most news events tend to have limited or substantial market influence.'
    )

with tab3:
    # seaborn count plots showing the distribution of sectors affected by news events, top 10 news souces, and top 10 news event types
    st.write('**Which sectors, sources, and event types are most commonly associated with financial news events?**')

    st.write(
        'This section examines how news events are distributed across market sectors, news sources, and event types. These'
        'categorical breakdowns help identify which areas of the market and which information channels appear most frequently'
        'in reported financial events.'
    )

    st.write(
        '**Note:** The visualisations in this tab reflect the currently selected date range. Adjusting the date filter may change'
          'the shape of these distributions, highlighting how news and market behaviour vary across different periods.'
    )

    fig, ax = plt.subplots(figsize=(12,6))
    sns.countplot(x='Sector', data=df_filtered, palette='Set2', edgecolor='black', linewidth=2, ax=ax)
    ax.set_title('Count of Sectors Affected by News Events')
    ax.set_ylabel('Count')
    ax.set_xlabel('Sectors')
    ax.grid(True, axis='y')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.write(
        'This chart shows the distribution of news events across different market sectors within the selected time period.'
        'Higher counts indicate sectors that are more frequently referenced in news coverage, which may reflect periods of'
        'heightened attention or activity.'
    )

    top_10_sources = df['Source'].value_counts().head(10).index
    fig, ax = plt.subplots(figsize=(12,6))
    sns.countplot(x='Source', data=df_filtered, palette='Set3', order=top_10_sources, edgecolor='black', linewidth=2, ax=ax)
    ax.set_title('Count of top 10 News Sources')
    ax.set_ylabel('Count')
    ax.set_xlabel('News Sources')
    ax.grid(True, axis='y')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.write(
        'This chart shows the distribution of news events by source. Differences in frequency may indicate which information'
        'channels are most active or prominent in reporting financial events during the selected period.'
    )

    top_10_market_events = df['Market_Event'].value_counts().head(10).index
    fig, ax = plt.subplots(figsize=(12,6))
    sns.countplot(x='Market_Event', data=df_filtered, palette='pastel', order=top_10_market_events, edgecolor='black', linewidth=2, ax=ax)
    ax.set_title('Count of top 10 Event Types')
    ax.set_ylabel('Count')
    ax.set_xlabel('Market Event')
    ax.grid(True, axis='y')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.write('This chart displays the most common types of market events reported in the dataset, helping to understand which'
            'events dominate the news landscape.')

import seaborn as sns

with tab4:
    #seaborn boxplots exploring how index percent chane and trading volume are affected by various factors
    st.write('**How do market returns and trading activity differ across news-related categories?**')

    st.write(
        'This section examines how market outcomes, specifically index percentage changes and trading volumes vary across different'
        'contextual factors such as event impact level, news sentiment, and market sector. Boxplots are used to compare distributions,'
        'spreads, and outliers, allowing assessment of whether certain categories are associated with systematically different market'
        'behaviour.'
    )

    st.write(
        '**Note:** The visualisations in this tab reflect the currently selected date range. Adjusting the date filter may change'
          'the shape of these distributions, highlighting how news and market behaviour vary across different periods.'
    )

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Impact_Level', y='Index_Change_Percent', data=df_filtered, ax=ax, palette='Set2', linewidth=2)
    ax.set_title('Index Change Percent by Impact Level')
    ax.set_ylabel('Index Change Percent')
    ax.set_xlabel('Impact Level')
    ax.grid(True, axis='y')
    st.pyplot(fig)

    st.write(
        'This chart compares the distribution of index percentage changes across low, medium, and high impact events within the'
        'selected time period. Differences in medians, spreads, or outliers may suggest variation in market responses across'
        'impact levels.'
    )

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Sentiment', y='Index_Change_Percent', data=df_filtered, ax=ax, palette='Set3', linewidth=2)
    ax.set_title('Index Change Percent by Sentiment')
    ax.set_ylabel('Index Change Percent')
    ax.set_xlabel('Sentiment')
    ax.grid(True, axis='y')
    st.pyplot(fig)

    st.write(
        'This chart compares index percentage changes across positive, neutral, and negative sentiment categories. Overlap between'
        'distributions indicates uncertainty, while shifts in medians may suggest sentiment-related differences in market response.'
    )

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Sector', y='Index_Change_Percent', data=df_filtered, ax=ax, palette='pastel', linewidth=2)
    ax.set_title('Index Change Percent by Sector')
    ax.set_ylabel('Index Change Percent')
    ax.set_xlabel('Sector')
    ax.grid(True, axis='y')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.write('This chart displays index changes for events affecting different sectors, revealing which sectors experience '
             'greater volatility or larger shifts in response to news during the selected period.')

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Impact_Level', y='Trading_Volume', data=df_filtered, ax=ax, palette='Set2', linewidth=2)
    ax.set_title('Trading Volume by Impact Level')
    ax.set_ylabel('Trading Volume')
    ax.set_xlabel('Impact Level')
    ax.grid(True, axis='y')
    st.pyplot(fig)

    st.write(
        'This chart shows how trading volume varies across impact levels. Wider spreads or higher medians may suggest increased'
        'market activity following higher-impact events.'
    )

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Sentiment', y='Trading_Volume', data=df_filtered, ax=ax, palette='Set3', linewidth=2)
    ax.set_title('Trading Volume by Sentiment')
    ax.set_ylabel('trading Volume')
    ax.set_xlabel('Sentiment')
    ax.grid(True, axis='y')
    st.pyplot(fig)

    st.write(
        'This chart displays trading volume distributions across sentiment categories. Differences in spread or central tendency'
        'may reflect varying levels of market engagement.'
    )

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Sector', y='Trading_Volume', data=df_filtered, ax=ax, palette='pastel', linewidth=2)
    ax.set_title('Trading Volume by Sector')
    ax.set_ylabel('Trading Volume')
    ax.set_xlabel('Sector')
    ax.grid(True, axis='y')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.write(
        'This chart shows how trading volume varies across sectors. Outliers and dispersion provide insight into sector-specific'
        'responses to news.'
    )

with tab5:
    #seaborn heatmap which explores correlations between key features, encoding was done in jupyter and heatmap_df was exported
    st.write('**How are key numerical features related across the full dataset?**')
    st.write(
        'This heatmap visualizes the correlations between trading sentiment, trading volume, index change percent and impact level.'
        'Correlations are computed across the entire dataset to highlight overall linear relationships that persist beyond specific'
        'time windows.'
    )
    
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(heatmap_df.corr(), annot=True, cmap='RdBu_r', center=0)
    ax.set_title("Correlation Heatmap of Key Financial Features")
    st.pyplot(fig)

    st.write(
        'The relationships in the heatmap are generally weak. The strongest correlation is only 0.037, found between sentiment and'
        'trading volume, suggesting that sentiment in headlines does not strongly align with trading behavior in this dataset.'
    )

with tab6:
        #most common word, bigrams, and trigrams are plotted using seaborn barplots, tokenisation, removal of punctuation and 
        #stopwords was carried out in jupyter notebooks
    
        from nltk import bigrams, trigrams, FreqDist

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
    

import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score

with tab7:
    #define classification model features and target
    features = ['Index_Change_Percent', 'Trading_Volume', 'Sentiment_Numeric', 
            'Sector_Aerospace & Defense', 'Sector_Agriculture', 'Sector_Automotive', 
            'Sector_Construction', 'Sector_Consumer Goods', 'Sector_Energy', 
            'Sector_Finance', 'Sector_Healthcare', 'Sector_Industrials', 
            'Sector_Materials', 'Sector_Media & Entertainment', 
            'Sector_Pharmaceuticals', 'Sector_Real Estate', 'Sector_Retail', 
            'Sector_Technology', 'Sector_Telecommunications', 'Sector_Transportation', 
            'Sector_Utilities', 'Market_Event_Bond Market Fluctuation', 
            'Market_Event_Central Bank Meeting', 'Market_Event_Commodity Price Shock', 
            'Market_Event_Consumer Confidence Report', 
            'Market_Event_Corporate Earnings Report', 'Market_Event_Cryptocurrency Regulation', 
            'Market_Event_Currency Devaluation', 'Market_Event_Economic Data Release', 
            'Market_Event_Geopolitical Event', 'Market_Event_Government Policy Announcement', 
            'Market_Event_IPO Launch', 'Market_Event_Inflation Data Release', 
            'Market_Event_Interest Rate Change', 'Market_Event_Major Merger/Acquisition', 
            'Market_Event_Market Rally', 'Market_Event_Regulatory Changes', 
            'Market_Event_Stock Market Crash', 'Market_Event_Supply Chain Disruption', 
            'Market_Event_Trade Tariffs Announcement', 
            'Market_Event_Unemployment Rate Announcement']
    x = df_one_hot_encoded[features]
    y = df_one_hot_encoded.Impact_Level_Numeric

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    dtree = DecisionTreeClassifier(max_depth=4)
    dtree = dtree.fit(x_train, y_train)

    st.write('**Can news and market features predict the impact level of financial events?**')
    
    st.write(
        'This section presents a decision tree classifier trained to predict the impact level of market events (Low, Medium, High)'
        'using numerical market indicators, sentiment measures, and encoded categorical features. The model is trained on the full'
        'dataset to identify global patterns rather than time-specific effects.'
    )

    st.write(
        'Model Features:\n'
        'The model uses a combination of numerical and one-hot encoded categorical features:'
        '''
-Market indicators: Index_Change_Percent, Trading_Volume
-Sentiment: Sentiment_Numeric
-Sector indicators: one-hot encoded sector variables (e.g., Technology, Healthcare)
-Market events: (e.g., IPO launches, bond market fluctuations)
        '''
    )  

    st.write(
        'How the Decision Tree Works\n'
        'Each internal node represents a split on a feature chosen to maximise class separation, using Gini impurity. The tree'
        'recursively partitions the data until stopping conditions are met. Leaf nodes represent the predicted impact level'
        '(Low, Medium, or High) for observations reaching that node.'
    )

    st.write(
        'Interpreting the Tree\n'
        'The root node corresponds to the feature providing the largest reduction in impurity and is therefore the most influential'
        'feature in the model. Paths from the root to a leaf represent a sequence of decision rules leading to a predicted impact'
        'level. Node colours indicate the class distribution at each node, with darker colours reflecting higher class purity.'
    )

    fig, ax = plt.subplots(figsize=(30,25))
    class_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
    class_labels = [class_mapping[c] for c in dtree.classes_]
    tree.plot_tree(dtree, feature_names=features, class_names=class_labels, filled=True, fontsize=15, rounded=True)
    st.pyplot(fig)
    print('Decision Tree Classes: ',dtree.classes_,'\n')


    y_test_predict = dtree.predict(x_test)

    #st.write('Confusion matrix tree: \n', confusion_matrix(y_test, y_test_predict), '\n')
    st.write('Accuracy: ', accuracy_score(y_test ,y_test_predict), '\n')
    st.write(metrics.classification_report(y_test, y_test_predict),'\n')

    # Feature importance chart
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': dtree.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(data=importance_df.head(10), x='Feature', y='Importance', palette='Blues_r', edgecolor='black', linewidth=2, ax=ax)
    ax.set_title('Top 10 Feature Importances')
    ax.set_ylabel('Importance')
    ax.set_xlabel('Feature')
    plt.xticks(rotation=90, ha='right')
    st.pyplot(fig)

    st.write(
        'This chart shows the relative importance of the top features used by the decision tree, measured by their contribution to'
        'impurity reduction. Higher importance values indicate features that play a larger role in the model’s decision-making'
        'process, though they do not imply causal influence.'
    )

    st.write(
        'Model Performance Summary\n'
        'The decision tree model shows limited ability to predict whether news events are of low, medium, or high impact. Efforts'
        'to improve performance such as using only the most influential features or training a random forest model yielded minimal'
        'improvement. This suggests that the current set of features does not provide enough information to reliably distinguish'
        'between impact levels. Additional feature engineering, alternative text representations, or time-dependent modelling'
        'approaches may be needed to better capture the relationship between financial news and market impact.'
    )




    


