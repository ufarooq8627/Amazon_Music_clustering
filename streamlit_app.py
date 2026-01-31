import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Music Galaxy Analyzer", page_icon="🎵", layout="wide")

# --- TITLE & INTRO ---
st.title("🎵 Amazon Music Clustering: The AI DJ")
st.markdown("""
Welcome to the Music Galaxy! 🌌
This app uses **Machine Learning (K-Means)** to group songs into genres automatically based on how they sound.
""")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # Load your final sorted file
        df = pd.read_csv('Final_Amazon_Music_Project.csv')
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("❌ File not found! Please make sure 'Final_Amazon_Music_Project.csv' is in the same folder.")
    st.stop()
else:
    st.success("✅ Data Loaded Successfully!")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Controls")
show_raw_data = st.sidebar.checkbox("Show Raw Data")

# --- GLOBAL VARIABLES ---
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# --- HELPER FUNCTION: GET CLUSTER PROFILES ---
def get_cluster_profiles(df):
    """
    Compares cluster averages to the global average to find distinct features.
    """
    global_means = df[features].mean()
    cluster_profiles = {}
    
    # Calculate means per cluster
    cluster_means = df.groupby('cluster_label')[features].mean()
    
    for cluster_id in cluster_means.index:
        # Find features that are significantly higher than the global average
        # We calculate the % difference
        diff = (cluster_means.loc[cluster_id] - global_means) / global_means.abs()
        
        # Sort by the biggest positive difference
        top_features = diff.sort_values(ascending=False).head(3)
        
        # Create a descriptive string (e.g., "High Instrumentalness, High Acousticness")
        desc = ", ".join([f"High {feat.capitalize()}" for feat in top_features.index])
        cluster_profiles[cluster_id] = desc
        
    return cluster_profiles, cluster_means

# Calculate profiles once
profiles, cluster_means_df = get_cluster_profiles(df)

# --- SECTION 1: OVERVIEW ---
if show_raw_data:
    st.subheader("🔍 Raw Data Preview")
    st.dataframe(df.head(10))

# --- SECTION 2: CLUSTER INTERPRETATION (The "DNA") ---
st.header("1. Decoding the Music DNA 🧬")
st.write("We analyzed the audio features to generate these profiles automatically:")

# Display the profiles in a nice way
cols = st.columns(len(profiles))
for i, (cluster_id, desc) in enumerate(profiles.items()):
    with cols[i % len(cols)]: # Wrap around if many clusters
        st.info(f"**Cluster {cluster_id}**\n\n{desc}")

# Interactive Bar Chart
st.write("### 📊 Visual Comparison")
fig_bar = px.bar(
    cluster_means_df.reset_index().melt(id_vars='cluster_label'), 
    x='variable', 
    y='value', 
    color='cluster_label',
    barmode='group',
    title="Feature Comparison by Cluster",
    labels={'value': 'Average Score', 'variable': 'Audio Feature'}
)
st.plotly_chart(fig_bar, use_container_width=True)

# --- SECTION 3: THE GALAXY MAP (PCA) ---
st.header("2. The Music Galaxy Map 🗺️")
st.write("We used PCA to squash the data into 2D so we can see the clusters.")

if st.button("🚀 Generate Galaxy Map"):
    with st.spinner("Crunching the numbers..."):
        # Scale and PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])
        
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(X_scaled)
        
        # Create plot data
        pca_df = pd.DataFrame(pca_data, columns=['x', 'y'])
        pca_df['Cluster'] = df['cluster_label'].astype(str)
        pca_df['Song'] = df['name_song']
        pca_df['Artist'] = df['name_artists']
        
        # Plot
        fig_pca = px.scatter(
            pca_df, 
            x='x', 
            y='y', 
            color='Cluster',
            hover_data=['Song', 'Artist'],
            title="PCA Visualization of Music Clusters",
            opacity=0.6
        )
        st.plotly_chart(fig_pca, use_container_width=True)

# --- SECTION 4: RECOMMENDATION ENGINE ---
st.header("3. Song Recommender 🎧")
st.write("Pick a Cluster to see the top songs in that vibe!")

# Create a list of options with the profile description
cluster_options = [f"Cluster {k}: {v}" for k, v in profiles.items()]
selected_option = st.selectbox("Select a Vibe:", cluster_options)

# Extract the Cluster ID number from the string
selected_cluster_id = int(selected_option.split(":")[0].replace("Cluster ", ""))

# Filter data
filtered_songs = df[df['cluster_label'] == selected_cluster_id]

# Display Stats
st.metric("Songs in this Cluster", len(filtered_songs))
st.caption(f"ℹ️ Analysis: This cluster is defined by **{profiles[selected_cluster_id]}**.")

st.write(f"🎶 Here are 5 random songs from **Cluster {selected_cluster_id}**:")
display_cols = ['name_song', 'name_artists', 'release_date', 'popularity_songs']
st.dataframe(filtered_songs[display_cols].sample(5))

# --- SECTION 5: ARTIST SEARCH TOOL ---
st.header("4. Artist Analysis Tool 🕵️‍♀️")
st.write("Find out which 'Music Team' your favorite artist belongs to.")

# 1. Get unique artist list
all_artists = sorted(df['name_artists'].astype(str).unique())

# 2. Create Dropdown
selected_artist = st.selectbox("Select an Artist:", all_artists)

if selected_artist:
    # 3. Filter data
    artist_data = df[df['name_artists'] == selected_artist]

    st.success(f"Analyzing {len(artist_data)} songs by {selected_artist}...")
    
    if not artist_data.empty:
        # Show most common cluster
        top_cluster = artist_data['cluster_label'].mode()[0]
        cluster_desc = profiles[top_cluster]
        
        col1, col2 = st.columns(2)
        col1.metric(label="Most Common Cluster", value=f"Cluster {top_cluster}")
        col2.metric(label="Vibe", value=cluster_desc)
        
        # Show songs
        st.write(f"Top songs by {selected_artist}:")
        st.dataframe(artist_data[['name_song', 'cluster_label', 'release_date']].head(10))
        
        # Mini chart
        st.write("Cluster Distribution:")
        st.bar_chart(artist_data['cluster_label'].value_counts())

# --- FOOTER ---
st.markdown("---")
st.write("Created by a Junior Data Scientist 🎓")