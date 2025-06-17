import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn

st.image("spotify2.png")
st.title("Spotify Top Hits From 2000-2019")
st.subheader("🎶 Can danceability be predicted?")

df = pd.read_csv("songs_normalize (1).csv")

st.sidebar.title("Spotify Top Hits")
page = st.sidebar.selectbox("Select Page",["Introduction 📘","Visualization 📊", "Automated Report 📑","Prediction 💡"])

if page == "Introduction 📘":
    st.title("Objectives")
    st.subheader("Our goal is to help musicians and artists explore the relationship between musical elements to see if Danceability can be predicted. By analyzing Spotify data from 2000 to 2019, the app aims to reveal potential correlations that can inform creative decisions and audience engagement strategies.")

    st.subheader("Check out some of the top hits over the years!")

    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display",5,20,5)
    st.dataframe(df.head(rows))

    st.markdown("##### Missing values")
    missing = df.isnull().sum()
    st.write(missing)

    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ you have missing values")

    st.markdown("##### 📈 Summary Statistics of all songs")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())

elif page == "Visualization 📊":

    st.subheader("Data Vizualization")

    st.subheader("Correlation Matrix")
    df_numeric = df.select_dtypes(include=np.number)

    fig_corr, ax_corr = plt.subplots(figsize=(18,14))
    # create the plot, in this case with seaborn 
    sns.heatmap(df_numeric.corr(),annot=True,fmt=".2f",cmap='coolwarm')
    ## render the plot in streamlit 
    st.pyplot(fig_corr)

    df = df.dropna(subset=["genre", "danceability", "popularity"])

   
    st.subheader("Bar Plot: Danceability by Genre")
    df = df.dropna(subset=["genre", "danceability", "popularity"])
    df["genre"] = df["genre"].astype(str)

    df = df[df["genre"] != "set()"]

    df["genre"] = df["genre"].str.split(", ")
    df = df.explode("genre")
    df["genre"] = df["genre"].str.strip()

    grouped = df.groupby("genre").mean(numeric_only=True)

    counts = df["genre"].value_counts()
    grouped["count"] = counts

    grouped = grouped[grouped["count"] >= 10]

    metric = st.selectbox("Pick something to compare by genre:", ["danceability", "popularity"])

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=grouped.index, y=grouped[metric], ax=ax, palette="Set2")
    ax.set_title(f"{metric.title()} by Genre")
    ax.set_xlabel("Genre")
    ax.set_ylabel(metric.title())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)

    st.subheader("Box Plot: Popularity by Valence")
    st.write("Valence levels tell us what mood (positive or negative) a song is likely to evoke. On a scale out of 1.0, a song with a valence level > 0.5 will be more likely to make people happy while a song < 0.5 will be more likely to make people sad.")


    valence_levels = [] 

    for x in df["valence"]: 
        if x >= 0.5:
            valence_levels.append("High")  
        else:
            valence_levels.append("Low") 

    fig, ax = plt.subplots(figsize=(15, 10)) 
    sns.boxplot(x=valence_levels, y=df["popularity"])
    ax.set_title("Popularity by Valence Level")
    ax.set_xlabel("Valence Level")
    ax.set_ylabel("Popularity")
    st.pyplot(fig)

    st.subheader("Explicit vs Not Explicit Popular Songs from 2000-2019")
    df_pie = df["explicit"].value_counts()
    labels = ["Not Explicit", "Explicit"]  
    fig1, ax1 = plt.subplots()
    ax1.pie(df_pie, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)
    

elif page == "Prediction 💡":

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score 
    from sklearn.metrics import mean_absolute_error

    df_encoded = pd.get_dummies(df, columns=["genre", "artist"], drop_first=True)

    X = df_encoded[[
        "tempo", "energy", "valence", "loudness", "speechiness",
        "liveness", "duration_ms", "instrumentalness", "key", "year", "mode", "popularity", "acousticness"
    ] + [col for col in df_encoded.columns if col.startswith("genre_") or col.startswith("artist_")]]  # Add encoded genre columns

    y = df_encoded["danceability"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae= mean_absolute_error(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
    ax.set_xlabel("Actual Danceability")
    ax.set_ylabel("Predicted Danceability")
    ax.set_title("Actual vs Predicted Danceability")
    ax.legend()

    st.pyplot(fig)

    st.write(f"R² Score: {r2:.3f}")
    st.write(f"Mean Absolute Error: {mae:.3f}")
    st.write("The MAE shows us that the predictions are, on average, 8.5% off from the true value")
    
    
