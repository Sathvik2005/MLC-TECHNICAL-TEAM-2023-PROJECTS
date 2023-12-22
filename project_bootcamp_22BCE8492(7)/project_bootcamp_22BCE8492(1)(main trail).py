# Load the dataset into the analysis environment.

import pandas as pd
dataset_path = '/kaggle/input/2023-youtube-most-viewed-top600/top_600_youtube_videos_2023.csv'

# Loading the dataset
df = pd.read_csv(dataset_path)

# Display of first few rows of the dataset
print(df.head())

# Data cleaning and preprocessing methodologies.

import pandas as pd

# Check for missing values in the dataset
print("Missing values in each column:\n", df.isnull().sum())

# Handle missing values
# Fill missing values in 'like_count' and 'comment_count' with 0
df['like_count'].fillna(0, inplace=True)
df['comment_count'].fillna(0, inplace=True)

# Convert 'published_at' to datetime format
df['published_at'] = pd.to_datetime(df['published_at'])

# Additional preprocessing steps can be added here if necessary

# Displaying the information about the dataset after preprocessing
print("\nDataset Information after Preprocessing:")
print(df.info())
# 3.1. Statistical Overview
# Providing basic descriptive statistics for key metrics in the dataset.

print("Descriptive Statistics:")
print(df.describe())
# 3.2. Distribution Analysis
# Visualizing the distribution of views, likes, and comments.

import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of Views
plt.figure(figsize=(10, 6))
sns.histplot(df['view_count'], kde=True)
plt.title('Distribution of View Counts')
plt.xlabel('View Count')
plt.ylabel('Frequency')
plt.show()

# Distribution of Likes
plt.figure(figsize=(10, 6))
sns.histplot(df['like_count'], kde=True)
plt.title('Distribution of Like Counts')
plt.xlabel('Like Count')
plt.ylabel('Frequency')
plt.show()

# Distribution of Comments
plt.figure(figsize=(10, 6))
sns.histplot(df['comment_count'], kde=True)
plt.title('Distribution of Comment Counts')
plt.xlabel('Comment Count')
plt.ylabel('Frequency')
plt.show()
# 3.2. Distribution Analysis
# Creating summary tables for the distributions of views, likes, and comments.

# Summary for View Count Distribution
view_count_summary = df['view_count'].describe()
print("View Count Distribution Summary:")
print(view_count_summary.to_string())

# Summary for Like Count Distribution
like_count_summary = df['like_count'].describe()
print("\nLike Count Distribution Summary:")
print(like_count_summary.to_string())

# Summary for Comment Count Distribution
comment_count_summary = df['comment_count'].describe()
print("\nComment Count Distribution Summary:")
print(comment_count_summary.to_string())
# 3.3. Top Performing Videos
# Analyzing videos with the highest views, likes, and comments.

# Top Videos by Views
top_views = df.nlargest(10, 'view_count')
print("Top 10 Videos by Views:")
print(top_views[['title', 'view_count']])

# Top Videos by Likes
top_likes = df.nlargest(10, 'like_count')
print("\nTop 10 Videos by Likes:")
print(top_likes[['title', 'like_count']])

# Top Videos by Comments
top_comments = df.nlargest(10, 'comment_count')
print("\nTop 10 Videos by Comments:")
print(top_comments[['title', 'comment_count']])
3.4. Time Series Analysis
# Examining trends in views, likes, and comments over time.

# Converting 'published_at' to just the date for easier analysis
df['publish_date'] = df['published_at'].dt.date

# Grouping data by publish date and summing numerical columns
date_group = df.groupby('publish_date')[['view_count', 'like_count', 'comment_count']].sum()

# Plotting trends over time
plt.figure(figsize=(12, 6))
plt.plot(date_group.index, date_group['view_count'], label='Views')
plt.plot(date_group.index, date_group['like_count'], label='Likes')
plt.plot(date_group.index, date_group['comment_count'], label='Comments')
plt.title('Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Counts')
plt.legend()
plt.xticks(rotation=45)
plt.show()
# Generating a summary table for the time series analysis

# Grouping data by publish date and summing numerical columns
date_group = df.groupby('publish_date')[['view_count', 'like_count', 'comment_count']].sum()

# Converting the groupby object to a DataFrame for a tabular view
time_series_summary = date_group.reset_index()
time_series_summary.columns = ['Publish Date', 'Total Views', 'Total Likes', 'Total Comments']

# Displaying the summary table
print("Time Series Analysis Summary:")
print(time_series_summary.to_string(index=False))
# 3.5. Comparative Analysis of View Counts Across Different Days of the Week
# Analyzing if specific days of the week tend to have higher view counts.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Extracting the day of the week from the publish date
df['day_of_week'] = df['publish_date'].apply(lambda x: x.strftime('%A'))

# Grouping data by day of the week and calculating the mean view count for each day
views_per_day_of_week = df.groupby('day_of_week')['view_count'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Plotting the average view counts for each day of the week
plt.figure(figsize=(10, 6))
sns.barplot(x=views_per_day_of_week.index, y=views_per_day_of_week.values)
plt.title('Average View Counts Across Different Days of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average View Count')
plt.show()
# 3.5. Comparative Analysis of View Counts Across Different Days of the Week
# Creating a table to compare average view counts across different days of the week.

import pandas as pd

# Extracting the day of the week from the publish date
df['day_of_week'] = df['publish_date'].apply(lambda x: x.strftime('%A'))

# Grouping data by day of the week and calculating the mean view count for each day
views_per_day_of_week = df.groupby('day_of_week')['view_count'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Creating a DataFrame for the table view
views_per_day_df = pd.DataFrame({'Day of the Week': views_per_day_of_week.index, 'Average View Count': views_per_day_of_week.values})

# Displaying the summary table
print("Average View Counts Across Different Days of the Week:")
print(views_per_day_df.to_string(index=False))
 #4.1. Content Trends
# Identifying trends in video titles and uploads.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to extract keywords or themes from titles
def extract_keywords(title):
    # Example: Extract keywords/themes. Customize based on observed trends in titles.
    keywords = ['tutorial', 'review', 'challenge', 'vlog', 'how to']
    for keyword in keywords:
        if keyword in title.lower():
            return keyword
    return 'other'

# Applying the function to create a new 'Keywords' column
df['Keywords'] = df['title'].apply(extract_keywords)

# Analyzing frequency of keywords/themes in video titles
keyword_distribution = df['Keywords'].value_counts()

# Plotting the distribution of keywords/themes
plt.figure(figsize=(10, 6))
sns.barplot(x=keyword_distribution.index, y=keyword_distribution.values)
plt.title('Frequency of Keywords/Themes in Video Titles')
plt.xlabel('Keywords')
plt.ylabel('Frequency')
plt.show()

# Analyzing upload patterns - distribution of videos over time
plt.figure(figsize=(10, 6))
df['publish_date'].value_counts().sort_index().plot(kind='line')
plt.title('Video Upload Patterns Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Videos Uploaded')
plt.show()
# 4.2. Engagement Analysis
# Examining correlations between views, likes, and comments.

# Calculating the correlation matrix
correlation_matrix = df[['view_count', 'like_count', 'comment_count']].corr()

# Plotting the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Views, Likes, and Comments')
plt.show()
# 4.2. Engagement Analysis
# Creating a table to summarize the correlations between views, likes, and comments.

# Calculating the correlation matrix
correlation_matrix = df[['view_count', 'like_count', 'comment_count']].corr()

# Converting the correlation matrix to a DataFrame for a tabular view
correlation_table = correlation_matrix.reset_index()
correlation_table.columns = ['Metric', 'View Count Correlation', 'Like Count Correlation', 'Comment Count Correlation']

# Displaying the correlation table
print("Correlation Between Views, Likes, and Comments:")
print(correlation_table.to_string(index=False))
# 5.1. Popularity Over Time - Month-by-Month Analysis
# Analyzing how video popularity varies with different months of the year.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Extracting month from the publish date
df['publish_month'] = df['publish_date'].apply(lambda x: x.strftime('%B'))

# Grouping data by month and calculating the mean view count for each month
views_per_month = df.groupby('publish_month')['view_count'].mean()

# Ordering the months properly
months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
views_per_month = views_per_month.reindex(months_order)

# Plotting the average view counts for each month
plt.figure(figsize=(12, 6))
sns.barplot(x=views_per_month.index, y=views_per_month.values)
plt.title('Average View Counts Across Different Months of the Year')
plt.xlabel('Month')
plt.ylabel('Average View Count')
plt.xticks(rotation=45)
plt.show()
# 5.2. Seasonal Patterns
# Identifying any seasonal or temporal trends in video popularity.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Creating a column for the season based on the month
def get_season(month):
    if month in ['December', 'January', 'February']:
        return 'Winter'
    elif month in ['March', 'April', 'May']:
        return 'Spring'
    elif month in ['June', 'July', 'August']:
        return 'Summer'
    else:
        return 'Autumn'

df['season'] = df['publish_month'].apply(get_season)

# Grouping data by season and calculating the mean view count for each season
views_per_season = df.groupby('season')['view_count'].mean()

# Ordering the seasons properly
seasons_order = ['Winter', 'Spring', 'Summer', 'Autumn']
views_per_season = views_per_season.reindex(seasons_order)

# Plotting the average view counts for each season
plt.figure(figsize=(10, 6))
sns.barplot(x=views_per_season.index, y=views_per_season.values)
plt.title('Average View Counts Across Different Seasons')
plt.xlabel('Season')
plt.ylabel('Average View Count')
plt.show()
# 6.1. Metrics Comparison
# Comparative analysis across different content metrics.

# Scatter plot to compare views and likes
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='view_count', y='like_count')
plt.title('Comparison of Views and Likes')
plt.xlabel('View Count')
plt.ylabel('Like Count')
plt.show()

# Scatter plot to compare views and comments
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='view_count', y='comment_count')
plt.title('Comparison of Views and Comments')
plt.xlabel('View Count')
plt.ylabel('Comment Count')
plt.show()
# 6.1. Metrics Comparison - Creating a Summary Table
# Generating a summary table to compare different content metrics like views, likes, and comments.

import pandas as pd

# Calculating key statistics for views, likes, and comments
views_stats = df['view_count'].describe()
likes_stats = df['like_count'].describe()
comments_stats = df['comment_count'].describe()

# Creating a summary DataFrame
metrics_comparison_summary = pd.DataFrame({
    'Metric': ['Views', 'Likes', 'Comments'],
    'Mean': [views_stats['mean'], likes_stats['mean'], comments_stats['mean']],
    'Median': [views_stats['50%'], likes_stats['50%'], comments_stats['50%']],
    'Std Dev': [views_stats['std'], likes_stats['std'], comments_stats['std']],
    'Max': [views_stats['max'], likes_stats['max'], comments_stats['max']]
})

# Displaying the summary table
print("Summary Table for Metrics Comparison:")
print(metrics_comparison_summary.to_string(index=False))
# Creating a 'Category' column in the DataFrame based on keywords in the video titles

# Function to categorize videos based on titles
def categorize_video(title):
    # Define your categories and corresponding keywords
    categories = {
        'Education': ['tutorial', 'lecture', 'education'],
        'Entertainment': ['vlog', 'comedy', 'entertainment'],
        'Music': ['music', 'song', 'concert'],
        # Add more categories as needed
    }
    
    for category, keywords in categories.items():
        if any(keyword in title.lower() for keyword in keywords):
            return category
    
    return 'Other'  # Default category

# Apply the function to create a new 'Category' column
df['Category'] = df['title'].apply(categorize_video)

# Now the DataFrame has a new column 'Category' which can be used for further analysis
# 6.2. Category Analysis
# Exploring viewer preferences across different video categories.

# Calculating the mean views for each category
category_views = df.groupby('Category')['view_count'].mean().sort_values(ascending=False)

# Bar plot for average views per category
plt.figure(figsize=(12, 6))
sns.barplot(x=category_views.values, y=category_views.index)
plt.title('Average Views per Video Category')
plt.xlabel('Average View Count')
plt.ylabel('Category')
plt.show()
# Creating a 'Category' column in the DataFrame based on keywords in the video titles

# Function to categorize videos based on titles
def categorize_video(title):
    # Define your categories and corresponding keywords
    categories = {
        'Education': ['tutorial', 'lecture', 'education'],
        'Entertainment': ['vlog', 'comedy', 'entertainment'],
        'Music': ['music', 'song', 'concert'],
        # Add more categories as needed
    }
    
    for category, keywords in categories.items():
        if any(keyword in title.lower() for keyword in keywords):
            return category
    
    return 'Other'  # Default category

# Apply the function to create a new 'Category' column
df['Category'] = df['title'].apply(categorize_video)

# Now the DataFrame has a new column 'Category' which can be used for further analysis
# 6.2. Category Analysis
# Exploring viewer preferences across different video categories.

# Calculating the mean views for each category
category_views = df.groupby('Category')['view_count'].mean().sort_values(ascending=False)

# Bar plot for average views per category
plt.figure(figsize=(12, 6))
sns.barplot(x=category_views.values, y=category_views.index)
plt.title('Average Views per Video Category')
plt.xlabel('Average View Count')
plt.ylabel('Category')
plt.show()
# Investigating the Distribution of Publishing Times for Entertainment Videos

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure 'publish_date' is in datetime format
df['publish_date'] = pd.to_datetime(df['publish_date'])

# Filtering for Entertainment category videos
entertainment_videos = df[df['Category'] == 'Entertainment'].copy()

# Extracting the publishing hour
entertainment_videos['publish_hour'] = entertainment_videos['publish_date'].dt.hour

# Checking the unique values and their counts in 'publish_hour'
publish_hour_counts = entertainment_videos['publish_hour'].value_counts()
print("Publish Hour Counts in Entertainment Videos:")
print(publish_hour_counts)

# Plotting the distribution if multiple hours are present
if len(publish_hour_counts) > 1:
    plt.figure(figsize=(12, 6))
    sns.barplot(x=publish_hour_counts.index, y=publish_hour_counts.values)
    plt.title('Distribution of Posting Times for Entertainment Videos')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Videos Posted')
    plt.xticks(range(0, 24))
    plt.show()
else:
    print("All or most Entertainment videos are published at the same hour.")
    #7.1 Sentiment Analysis of Video Titles
# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Create a new column for sentiment polarity
df['title_sentiment'] = df['title'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Plotting the distribution of title sentiment
plt.figure(figsize=(10, 6))
plt.hist(df['title_sentiment'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Sentiment in Video Titles')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Number of Videos')
plt.show()

# Grouping by category and calculating the average sentiment for each category
category_sentiment = df.groupby('Category')['title_sentiment'].mean().sort_values(ascending=False)

# Bar plot for average sentiment per category
plt.figure(figsize=(12, 6))
category_sentiment.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Average Sentiment in Video Titles by Category')
plt.xlabel('Category')
plt.ylabel('Average Sentiment Polarity')
plt.xticks(rotation=45)
plt.show()
# Creating a summary table for sentiment analysis results

# Grouping by category and calculating the average sentiment for each category
category_sentiment = df.groupby('Category')['title_sentiment'].mean()

# Converting the groupby object to a DataFrame for a tabular view
sentiment_summary = pd.DataFrame({'Category': category_sentiment.index, 'Average Sentiment': category_sentiment.values})

# Sorting the DataFrame by average sentiment
sentiment_summary = sentiment_summary.sort_values(by='Average Sentiment', ascending=False)

# Displaying the summary table
print("Average Sentiment in Video Titles by Category:")
print(sentiment_summary.to_string(index=False))
# 7.2 Predictive Modeling for Video Popularity with Random Forest Regressor

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

# Ensure required columns exist
required_columns = ['title_sentiment', 'duration_seconds', 'view_count']
if not all(column in df.columns for column in required_columns):
    print("Required columns are missing from the dataset. Please check the dataset.")
else:
    # Selecting features for the model
    features = ['title_sentiment', 'duration_seconds']

    # Preparing the data for modeling
    X = df[features]
    y = df['view_count']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating and training the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predicting view counts on the test set
    y_pred_rf = rf_model.predict(X_test)

    # Evaluating the model
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    print(f'Mean Squared Error with Random Forest: {mse_rf}')

