import streamlit as st
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def render(df_one_hot_encoded):
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
        """
        **Model Features:**
        The model uses a combination of numerical and one-hot encoded categorical features:
        - Market indicators: Index_Change_Percent, Trading_Volume
        - Sentiment: Sentiment_Numeric
        - Sector indicators: one-hot encoded sector variables (e.g., Technology, Healthcare)
        - Market events: (e.g., IPO launches, bond market fluctuations)
        """
    )

    st.write(
        '**How the Decision Tree Works:**\n\n'
        'Each internal node represents a split on a feature chosen to maximise class separation, using Gini impurity. The tree'
        'recursively partitions the data until stopping conditions are met. Leaf nodes represent the predicted impact level'
        '(Low, Medium, or High) for observations reaching that node.'
    )

    st.write(
        '**Interpreting the Tree:**\n\n'
        'The root node corresponds to the feature providing the largest reduction in impurity. '
        'Node colors indicate the class distribution, with darker colors reflecting higher purity.'
    )

    fig, ax = plt.subplots(figsize=(30,25))
    class_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
    class_labels = [class_mapping[c] for c in dtree.classes_]
    tree.plot_tree(dtree, feature_names=features, class_names=class_labels, filled=True, fontsize=15, rounded=True)
    st.pyplot(fig)
    print('Decision Tree Classes: ',dtree.classes_,'\n')


    y_test_predict = dtree.predict(x_test)

    st.write('Confusion matrix tree: \n', confusion_matrix(y_test, y_test_predict), '\n')
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
        'process, though they do not imply causal influence.\n\n'
    )

    st.write(
        """
    **Model Performance Summary:**
    
    The decision tree model shows limited ability to predict whether news events are of low, medium, or high impact. 
    Efforts to improve performance, such as using only the most influential features or training a random forest model yielded minimal improvement. 
    
    This suggests that the current set of features does not provide enough information to reliably distinguish between impact levels. 
    Additional feature engineering, alternative text representations, or time-dependent modelling approaches may be needed to better capture the relationship between financial news and market impact.
    """
    )