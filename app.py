# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:57:22 2024

@author: GHRCE
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/processed_survey.csv')  # Replace with the correct path to your processed dataset
results = pd.read_csv('data/model_resultsmain.csv')  # Replace with the path to your model evaluation results

st.title("@@@College Survey Sentiment Analysis Dashboard@@@")

# # Section 1: Overall Sentiment Distribution
# st.header("Overall Sentiment Distribution")
# sentiment_counts = df['sentiment'].value_counts()
# st.bar_chart(sentiment_counts)

# Section: Overall Sentiment Distribution with a Pie Chart
st.header("Overall Sentiment Distribution ")

# Calculate sentiment counts
sentiment_counts = df['sentiment'].value_counts()
st.write("Sentiment Counts:", sentiment_counts)
# Check for the existence of "Happy" and "Unhappy" in sentiment counts
happy_count = sentiment_counts.get('happy', 0)  # Default to 0 if not present
unhappy_count = sentiment_counts.get('unhappy', 0)
  # Default to 0 if not present
st.write(f"Happy Count: {happy_count}, Unhappy Count: {unhappy_count}")
# Display a concern message if "Unhappy" responses exceed "Happy" responses
if unhappy_count > happy_count:
    st.warning("⚠️ Concern: The number of 'Unhappy' responses exceeds the number of 'Happy' responses!")
elif happy_count > 0 and unhappy_count == 0:
    st.success("😊 Great News: There are only 'Happy' responses with no 'Unhappy' responses!")
elif happy_count == 0 and unhappy_count > 0:
    st.warning("⚠️ Alert: No 'Happy' responses were recorded, but there are 'Unhappy' responses.")

# Define a color palette for the pie chart
colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99']  # Customize these colors

# Create a pie chart
fig, ax = plt.subplots()
ax.pie(
    sentiment_counts, 
    labels=sentiment_counts.index, 
    autopct='%1.1f%%', 
    startangle=90, 
    colors=colors
)
ax.axis('equal')  # Ensures the pie chart is a perfect circle

# Display the chart in Streamlit
st.pyplot(fig)

# # Section 2: Department-wise Sentiment
# st.header("Department-wise Sentiment Analysis")
# department_sentiment = df.groupby('department')['sentiment_label'].mean()  # Assuming 'sentiment_label' is numeric
# st.bar_chart(department_sentiment)

# # Section 2: Department-wise Sentiment Analysis
# st.header("Department-wise Sentiment Analysis")

# # Group by department and sentiment, then count occurrences
# department_sentiment_counts = df.groupby(['department', 'sentiment']).size().unstack(fill_value=0)

# # Plot the bar chart
# st.bar_chart(department_sentiment_counts)

# Section 2: Department-wise Sentiment Analysis
st.header("Department-wise Sentiment Analysis")

# Group by department and sentiment, then count occurrences
department_sentiment_counts = df.groupby(['department', 'sentiment']).size().unstack(fill_value=0)
# Check for departments where 'Unhappy' count exceeds 'Happy' count
concern_departments = department_sentiment_counts[
    (department_sentiment_counts.get('unhappy', 0) > department_sentiment_counts.get('happy', 0))
].index.tolist()

# Display the list of concerning departments
if concern_departments:
    st.warning("⚠️ Concern: The following departments have more 'Unhappy' responses than 'Happy':")
    for department in concern_departments:
        st.write(f"- {department}")
else:
    st.success("😊 Great News: No department has more 'Unhappy' responses than 'Happy'.")
# Plot the bar chart using Matplotlib
fig, ax = plt.subplots(figsize=(10, 6))
department_sentiment_counts.plot(kind='bar', stacked=False, ax=ax, color=['green', 'blue', 'red'])

# Customize the chart
ax.set_title("Department-wise Sentiment Analysis", fontsize=16)
ax.set_xlabel("Department", fontsize=14)
ax.set_ylabel("Sentiment Count", fontsize=14)
ax.legend(title="Sentiment", fontsize=12)
plt.xticks(rotation=45, fontsize=12)

# Display the chart in Streamlit
st.pyplot(fig)

# 
# Section 3: Facility-wise Sentiment Analysis
st.header("Facility-wise Sentiment Analysis")

# Group by category (facility) and sentiment, then count occurrences
facility_sentiment_counts = df.groupby(['category', 'sentiment']).size().unstack(fill_value=0)
# Check for facilities where 'Unhappy' count exceeds 'Happy' count
concern_facilities = facility_sentiment_counts[
    (facility_sentiment_counts.get('unhappy', 0) > facility_sentiment_counts.get('happy', 0))
].index.tolist()

# Display the list of concerning facilities, each on a new line
if concern_facilities:
    st.warning("⚠️ Concern: The following facilities have more 'Unhappy' responses than 'Happy':")
    for facility in concern_facilities:
        st.write(f"- {facility}")
else:
    st.success("😊 Great News: No facility has more 'Unhappy' responses than 'Happy'.")
# Plot the bar chart using Matplotlib
fig, ax = plt.subplots(figsize=(12, 6))
facility_sentiment_counts.plot(kind='bar', stacked=False, ax=ax, color=['green', 'blue', 'red'])

# Customize the chart
ax.set_title("Facility-wise Sentiment Analysis", fontsize=16)
ax.set_xlabel("Facility", fontsize=14)
ax.set_ylabel("Sentiment Count", fontsize=14)
ax.legend(title="Sentiment", fontsize=12)
plt.xticks(rotation=45, fontsize=12)

# Display the chart in Streamlit
st.pyplot(fig)
# Section 4: Model Comparative Analysis
st.header("Model Comparative Analysis")
st.dataframe(results)



# # Plot the bar chart for Accuracy and F1-Score
# st.header("Model Performance Comparison")

# # Define the color palette
# colors = sns.color_palette("Set2", len(results))  # Automatically assigns a unique color for each model

# # Create the bar chart
# fig, ax = plt.subplots(figsize=(12, 6))
# x = range(len(results))  # X-axis positions for the models

# # Plot accuracy bars
# ax.bar(x, results['Accuracy'], color=colors, width=0.4, label='Accuracy', align='center')

# # Plot F1-Score bars (shifted slightly for better visibility)
# ax.bar([i + 0.4 for i in x], results['F1-Score'], color=colors, width=0.4, alpha=0.7, label='F1-Score', align='center')

# # Add labels and formatting
# ax.set_xticks([i + 0.2 for i in x])
# ax.set_xticklabels(results['Model'], rotation=45)
# ax.set_ylabel("Performance Metrics")
# ax.set_title("Model Comparative Analysis")
# ax.legend()


# #main
# st.header("Model Performance Comparison")

# # Define the color palette
# colors = sns.color_palette("Set2", len(results))  # Automatically assigns unique colors for each model

# # Create the bar chart
# fig, ax = plt.subplots(figsize=(14, 8))
# x = range(len(results))  # X-axis positions for the models

# # Plot accuracy bars
# ax.bar(x, results['Accuracy'], color=colors, width=0.2, label='Accuracy', align='center')

# # Plot F1-Score bars (shifted slightly for better visibility)
# ax.bar([i + 0.2 for i in x], results['F1-Score'], color=colors, width=0.2, alpha=0.7, label='F1-Score', align='center')

# # Plot Precision bars (shifted further for visibility)
# ax.bar([i + 0.4 for i in x], results['Precision'], color=colors, width=0.2, alpha=0.7, label='Precision', align='center')

# # Plot Recall bars (shifted further for visibility)
# ax.bar([i + 0.6 for i in x], results['Recall'], color=colors, width=0.2, alpha=0.7, label='Recall', align='center')


# # Add labels and formatting
# ax.set_xticks([i + 0.3 for i in x])  # Adjust tick positions to align with the grouped bars
# ax.set_xticklabels(results['Model'], rotation=45)
# ax.set_ylabel("Performance Metrics")
# ax.set_title("Model Comparative Analysis")
# ax.legend()
# # Display the plot
# st.pyplot(fig)



# mainnext
st.header("Model Performance Comparison")

# Define the new color palette
colors = sns.color_palette("coolwarm", 4)  # 4 unique colors for Accuracy, F1-Score, Precision, and Recall

# Create the bar chart
fig, ax = plt.subplots(figsize=(14, 8))
x = range(len(results))  # X-axis positions for the models

# Plot accuracy bars
ax.bar(x, results['Accuracy'], color=colors[0], width=0.2, label='Accuracy', align='center')

# Plot F1-Score bars (shifted slightly for better visibility)
ax.bar([i + 0.2 for i in x], results['F1-Score'], color=colors[1], width=0.2, alpha=0.8, label='F1-Score', align='center')

# Plot Precision bars (shifted further for visibility)
ax.bar([i + 0.4 for i in x], results['Precision'], color=colors[2], width=0.2, alpha=0.8, label='Precision', align='center')

# Plot Recall bars (shifted further for visibility)
ax.bar([i + 0.6 for i in x], results['Recall'], color=colors[3], width=0.2, alpha=0.8, label='Recall', align='center')

# Add labels and formatting
ax.set_xticks([i + 0.3 for i in x])  # Adjust tick positions to align with the grouped bars
ax.set_xticklabels(results['Model'], rotation=45)
ax.set_ylabel("Performance Metrics")
ax.set_title("Model Comparative Analysis")
ax.legend()

# Display the plot
st.pyplot(fig)


# Display the chart in Streamlit
# st.pyplot(fig)

# # Section: Model Performance Comparison (Accuracy, F1-Score, Recall, Precision)
# st.header("Model Performance Comparison")

# # Assuming the 'results' DataFrame contains columns: Model, Accuracy, F1-Score, Precision, Recall
# # Set the model names as index
# results.set_index('Model', inplace=True)

# # Plot the bar chart using Matplotlib
# fig, ax = plt.subplots(figsize=(12, 6))

# # Plotting all metrics (Accuracy, F1-Score, Precision, Recall) side by side
# results[['Accuracy', 'F1-Score', 'Precision', 'Recall']].plot(kind='bar', ax=ax, color=['green', 'blue', 'red', 'purple'])

# # Customize the chart
# ax.set_title("Model Performance Comparison", fontsize=16)
# ax.set_xlabel("Model", fontsize=14)
# ax.set_ylabel("Score", fontsize=14)
# ax.legend(title="Metrics", fontsize=12)
# plt.xticks(rotation=45, fontsize=12)

# # Display the chart in Streamlit
# st.pyplot(fig)
