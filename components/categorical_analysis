import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def render(df_filtered, df):
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