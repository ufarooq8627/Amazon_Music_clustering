import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import os

#   PAGE CONFIG        
st.set_page_config(
    page_title="Music Galaxy Analyzer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

#   CUSTOM CSS        ─
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 2rem;
}

.cluster-card {
    background: linear-gradient(135deg, #1e1b4b22, #312e8122);
    border: 1px solid #4f46e533;
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    height: 100%;
}

.predict-box {
    background: linear-gradient(135deg, #0f172a, #1e1b4b);
    border: 2px solid #6d28d9;
    border-radius: 18px;
    padding: 2rem;
    margin: 1rem 0;
}

.result-banner {
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 1rem;
}

.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #c4b5fd;
    border-left: 4px solid #7c3aed;
    padding-left: 0.8rem;
    margin: 2rem 0 1rem;
}

div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    border: 1px solid #4f46e5;
    border-radius: 12px;
    padding: 0.8rem 1rem;
}
</style>
""", unsafe_allow_html=True)

#   CONSTANTS         
FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'duration_ms', 'valence', 'tempo'
]

# Cluster personality labels (generated from notebook analysis)
CLUSTER_VIBES = {
    0: ("🎸 Energetic Rockers",    "#f97316"),
    1: ("🎹 Ambient Dreamers",     "#60a5fa"),
    2: ("🕺 Dance Floor Bangers",  "#a78bfa"),
    3: ("🎤 Vocal Storytellers",   "#34d399"),
    4: ("🎷 Chill Instrumentals",  "#f472b6"),
}

#   DATA & MODEL LOADING       
@st.cache_data
def load_data():
    path = "Final_Amazon_Music_Project.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

@st.cache_resource
def load_models():
    if not (os.path.exists("kmeans_model.pkl") and os.path.exists("scaler.pkl")):
        return None, None
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return kmeans, scaler

df       = load_data()
kmeans, scaler = load_models()

#   HEADER 
st.markdown('<div class="main-title">🎵 Music Galaxy Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Amazon Music Clustering — AI-powered genre discovery</div>', unsafe_allow_html=True)

#   GUARD RAILS        
if df is None:
    st.error("❌ `Final_Amazon_Music_Project.csv` not found. Place it in the same folder.")
    st.stop()

if kmeans is None or scaler is None:
    st.warning("⚠️ Model files not found. Run `python train_model.py` first.")
    st.stop()

#   SIDEBAR         ─
st.sidebar.header("🎛️ Controls")
show_raw_data = st.sidebar.checkbox("Show Raw Data Preview")
st.sidebar.success(f"Dataset: **{len(df):,}** songs")
st.sidebar.info(f"Clusters: **{kmeans.n_clusters}**")

#   HELPER: Cluster profiles from data      
def get_cluster_profiles(df):
    # Exclude duration_ms from display — its 200k+ values crush 0-1 features in charts
    display_features = [f for f in FEATURES if f != 'duration_ms']
    global_means     = df[display_features].mean()
    cluster_means    = df.groupby('cluster_label')[display_features].mean()
    profiles = {}
    for cid in cluster_means.index:
        diff      = (cluster_means.loc[cid] - global_means) / global_means.abs()
        top_feats = diff.sort_values(ascending=False).head(3)
        desc      = ", ".join([f"High {f.capitalize()}" for f in top_feats.index])
        profiles[cid] = desc
    return profiles, cluster_means   # cluster_means already has duration_ms excluded

profiles, cluster_means_df = get_cluster_profiles(df)

#  
#  SECTION 0 — Raw Data
#  
if show_raw_data:
    st.markdown('<div class="section-header">🔍 Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

#  
#  SECTION 1 — Cluster DNA
#  
st.markdown('<div class="section-header">1. Decoding the Music DNA 🧬</div>', unsafe_allow_html=True)
st.write("Each cluster was automatically profiled by comparing its average audio features to the global average:")

cols = st.columns(5)
for i, cid in enumerate(sorted(profiles.keys())):
    vibe_label, vibe_color = CLUSTER_VIBES.get(cid, (f"Cluster {cid}", "#94a3b8"))
    with cols[i]:
        st.markdown(f"""
        <div style="border: 2px solid {vibe_color}; border-radius: 14px; padding: 1.1rem;
                    text-align: center; background: {vibe_color}18;">
            <div style="font-size:2rem; line-height:1;">{vibe_label.split()[0]}</div>
            <div style="color:{vibe_color}; font-weight:700; margin:0.4rem 0;
                        font-size:0.95rem;">{' '.join(vibe_label.split()[1:])}</div>
            <div style="font-size:0.72rem; color:#555; margin-top:0.3rem;">{profiles[cid]}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# Normalise each feature column to 0–1 so all features are on the same scale
# (tempo ~115, loudness ~-9 are out of range vs. 0-1 features otherwise)
plot_df = cluster_means_df.copy()
for col in plot_df.columns:
    col_min, col_max = df[col].min(), df[col].max()
    if col_max != col_min:
        plot_df[col] = (plot_df[col] - col_min) / (col_max - col_min)

vibe_colors_ordered = [CLUSTER_VIBES[cid][1] for cid in sorted(plot_df.index)]

melted_df = plot_df.reset_index().melt(id_vars='cluster_label')
# Cast to string so Plotly treats clusters as discrete categories (not a continuous colorscale)
melted_df['cluster_label'] = melted_df['cluster_label'].astype(str)
vibe_map = {str(cid): CLUSTER_VIBES[cid][1] for cid in sorted(plot_df.index)}

fig_bar = px.bar(
    melted_df,
    x='variable', y='value', color='cluster_label',
    barmode='group',
    title="Feature Comparison by Cluster (Normalised 0–1)",
    labels={'value': 'Normalised Score (0–1)', 'variable': 'Audio Feature', 'cluster_label': 'Cluster'},
    color_discrete_map=vibe_map
)
fig_bar.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color='#333',
    xaxis=dict(gridcolor='#e2e8f0', tickangle=-30),
    yaxis=dict(gridcolor='#e2e8f0', range=[0, 1]),
    legend_title_text='Cluster'
)
st.plotly_chart(fig_bar, use_container_width=True)

#  
#  SECTION 2 — Galaxy Map
#  
st.markdown('<div class="section-header">2. The Music Galaxy Map 🗺️</div>', unsafe_allow_html=True)
st.write("PCA reduces the 10 audio features to 2D so we can visualise all clusters at once.")

if st.button("🚀 Generate Galaxy Map"):
    with st.spinner("Crunching numbers..."):
        X_scaled_viz = scaler.transform(df[FEATURES])
        pca          = PCA(n_components=2)
        coords       = pca.fit_transform(X_scaled_viz)
        pca_df       = pd.DataFrame(coords, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = df['cluster_label'].astype(str)
        pca_df['Song']    = df['name_song']
        pca_df['Artist']  = df['name_artists']

        fig_pca = px.scatter(
            pca_df, x='PC1', y='PC2',
            color='Cluster',
            hover_data=['Song', 'Artist'],
            title="PCA Visualisation of Music Clusters",
            opacity=0.5,
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_pca.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
        )
        st.plotly_chart(fig_pca, use_container_width=True)

#  SECTION 3 — ✨ PREDICT MY SONG'S CLUSTER ✨  (NEW FEATURE)

st.markdown('<div class="section-header">3. Predict My Song\'s Cluster 🤖</div>', unsafe_allow_html=True)
st.write("""
Enter the **audio features** of any song and the trained **KMeans model** will predict 
which cluster it belongs to — exactly how it was trained in the notebook!
""")

st.markdown('<div class="predict-box">', unsafe_allow_html=True)

#   Song Name Input 
song_name = st.text_input(
    "🎵 Song Name",
    placeholder="e.g. Blinding Lights, Shape of You, Tum Hi Ho ...",
    help="Enter any song name — it will appear in the prediction result.",
    key="song_name_input"
)
display_name = song_name.strip() if song_name.strip() else "Your Song"

st.markdown("---")
st.subheader("🎚️ Set Audio Features")

# Feature min/max/default from dataset
feature_config = {
    'danceability':     dict(min=0.0,   max=1.0,     step=0.01, default=float(df['danceability'].median()),
                             help="How suitable for dancing (0=not at all, 1=very danceable)"),
    'energy':           dict(min=0.0,   max=1.0,     step=0.01, default=float(df['energy'].median()),
                             help="Intensity and activity (0=calm, 1=very energetic)"),
    'loudness':         dict(min=-60.0, max=0.0,     step=0.1,  default=float(df['loudness'].median()),
                             help="Overall loudness in dB (usually -60 to 0)"),
    'speechiness':      dict(min=0.0,   max=1.0,     step=0.01, default=float(df['speechiness'].median()),
                             help="Presence of spoken words (>0.66 = pure speech)"),
    'acousticness':     dict(min=0.0,   max=1.0,     step=0.01, default=float(df['acousticness'].median()),
                             help="Confidence the track is acoustic (0=electric, 1=acoustic)"),
    'instrumentalness': dict(min=0.0,   max=1.0,     step=0.01, default=float(df['instrumentalness'].median()),
                             help="Predicts whether a track has no vocals (>0.5 = instrumental)"),
    'liveness':         dict(min=0.0,   max=1.0,     step=0.01, default=float(df['liveness'].median()),
                             help="Probability of a live audience in the recording"),
    'duration_ms':      dict(min=30000, max=900000,  step=1000, default=int(df['duration_ms'].median()),
                             help="Song duration in milliseconds (e.g. 200000 = ~3m 20s)"),
    'valence':          dict(min=0.0,   max=1.0,     step=0.01, default=float(df['valence'].median()),
                             help="Musical positiveness (0=sad/angry, 1=happy/euphoric)"),
    'tempo':            dict(min=30.0,  max=250.0,   step=0.5,  default=float(df['tempo'].median()),
                             help="Estimated tempo in BPM"),
}

# Layout sliders in two rows of 5
row1_features = FEATURES[:5]
row2_features = FEATURES[5:]

user_input = {}

r1_cols = st.columns(5)
for col, feat in zip(r1_cols, row1_features):
    cfg = feature_config[feat]
    with col:
        user_input[feat] = st.slider(
            feat.replace('_', ' ').title(),
            min_value=cfg['min'],
            max_value=cfg['max'],
            value=cfg['default'],
            step=cfg['step'],
            help=cfg['help'],
            key=f"slider_{feat}"
        )

r2_cols = st.columns(5)
for col, feat in zip(r2_cols, row2_features):
    cfg = feature_config[feat]
    with col:
        user_input[feat] = st.slider(
            feat.replace('_', ' ').title(),
            min_value=cfg['min'],
            max_value=cfg['max'],
            value=cfg['default'],
            step=cfg['step'],
            help=cfg['help'],
            key=f"slider_{feat}"
        )

st.markdown('</div>', unsafe_allow_html=True)

#   PREDICT BUTTON         
predict_col, _, _ = st.columns([1, 2, 2])
with predict_col:
    predict_clicked = st.button("🔮  Predict Cluster", use_container_width=True, type="primary")

if predict_clicked:
    # Build input vector in the SAME feature order as training
    input_vector = np.array([[user_input[f] for f in FEATURES]])

    # Scale using the SAME scaler fitted on training data
    input_scaled = scaler.transform(input_vector)

    # Predict cluster
    predicted_cluster = int(kmeans.predict(input_scaled)[0])

    # Distance to ALL cluster centers (for confidence gauge)
    distances = np.linalg.norm(kmeans.cluster_centers_ - input_scaled, axis=1)
    closest   = distances[predicted_cluster]
    proba_like = 1 / (1 + distances)
    confidence = proba_like[predicted_cluster] / proba_like.sum() * 100

    vibe_label, vibe_color = CLUSTER_VIBES.get(predicted_cluster, (f"Cluster {predicted_cluster}", "#94a3b8"))

    st.success("Prediction complete!")

    #   Result Banner       
    st.markdown(f"""
    <div class="result-banner" style="background: linear-gradient(135deg, {vibe_color}22, {vibe_color}44);
         border: 2px solid {vibe_color}; color: {vibe_color};">
        <div style="font-size:1.1rem; color:#cbd5e1; margin-bottom:0.4rem;">🎵 <em>{display_name}</em></div>
        belongs to &nbsp;<strong>Cluster {predicted_cluster}</strong>&nbsp; — {vibe_label}<br>
        <span style="font-size:0.9rem; font-weight:400; color:#e2e8f0;">
            Audio Profile: {profiles[predicted_cluster]}
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    #   Stats row         
    m1, m2, m3 = st.columns(3)
    songs_in_cluster = len(df[df['cluster_label'] == predicted_cluster])
    m1.metric("Predicted Cluster",   f"Cluster {predicted_cluster}")
    m2.metric("Songs in this Cluster", f"{songs_in_cluster:,}")
    m3.metric("Model Confidence",    f"{confidence:.1f}%")

    #   Radar chart: user input vs cluster average    ─
    radar_features = [f for f in FEATURES if f != 'duration_ms']
    cluster_avg    = df[df['cluster_label'] == predicted_cluster][radar_features].mean()

    # Normalise user values to [0, 1] for radar (loudness needs special treatment)
    def normalise(feat, val):
        lo = float(df[feat].min())
        hi = float(df[feat].max())
        return (val - lo) / (hi - lo) if hi != lo else 0.5

    user_norm    = [normalise(f, user_input[f])       for f in radar_features]
    cluster_norm = [normalise(f, cluster_avg[f])      for f in radar_features]
    labels       = [f.replace('_', ' ').capitalize()  for f in radar_features]

    # Convert hex vibe_color to rgba for Plotly compatibility
    def hex_to_rgba(hex_color, alpha=0.2):
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=user_norm + [user_norm[0]],
        theta=labels + [labels[0]],
        fill='toself',
        name='Your Song',
        line_color='#a78bfa',
        fillcolor='rgba(167,139,250,0.2)'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=cluster_norm + [cluster_norm[0]],
        theta=labels + [labels[0]],
        fill='toself',
        name=f'Cluster {predicted_cluster} Avg',
        line_color=vibe_color,
        fillcolor=hex_to_rgba(vibe_color, 0.2)
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 1], color='#475569'),
            angularaxis=dict(color='#94a3b8'),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e2e8f0',
        title=f"\"{display_name}\" vs Cluster {predicted_cluster} Average",
        showlegend=True,
        legend=dict(font=dict(color='#e2e8f0'))
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    #   Similar songs from the predicted cluster     
    st.subheader(f"🎶 Songs similar to \"{display_name}\" — Cluster {predicted_cluster}")
    st.caption(f"These are real songs from your dataset that share the same cluster as {display_name}.")
    similar = df[df['cluster_label'] == predicted_cluster]
    display_cols = ['name_song', 'name_artists', 'genres', 'release_date', 'popularity_songs']
    st.dataframe(similar[display_cols].sample(min(10, len(similar))), use_container_width=True)


#  
#  SECTION 4 — Song Recommender
#  
st.markdown('<div class="section-header">4. Song Recommender 🎧</div>', unsafe_allow_html=True)
st.write("Pick a cluster vibe and discover songs that match it!")

cluster_options = [f"Cluster {k} — {CLUSTER_VIBES[k][0]}" for k in sorted(profiles.keys())]
selected_option = st.selectbox("Select a Vibe:", cluster_options, key="recommender_select")
selected_cluster_id = int(selected_option.split("—")[0].replace("Cluster ", "").strip())

filtered_songs = df[df['cluster_label'] == selected_cluster_id]
_, color = CLUSTER_VIBES.get(selected_cluster_id, ("", "#94a3b8"))

c1, c2 = st.columns(2)
c1.metric("Songs in Cluster", f"{len(filtered_songs):,}")
c2.metric("Cluster Vibe",     CLUSTER_VIBES[selected_cluster_id][0])

st.caption(f"Audio Profile: **{profiles[selected_cluster_id]}**")
st.dataframe(
    filtered_songs[['name_song', 'name_artists', 'genres', 'release_date', 'popularity_songs']].sample(8),
    use_container_width=True
)

#  
#  SECTION 5 — Artist Analysis
#  
st.markdown('<div class="section-header">5. Artist Analysis Tool 🕵️</div>', unsafe_allow_html=True)
st.write("Find out which cluster your favourite artist belongs to.")

all_artists     = sorted(df['name_artists'].astype(str).unique())
selected_artist = st.selectbox("Select an Artist:", all_artists, key="artist_select")

if selected_artist:
    artist_data = df[df['name_artists'] == selected_artist]
    st.success(f"Analysing **{len(artist_data)}** songs by **{selected_artist}**")

    top_cluster  = int(artist_data['cluster_label'].mode()[0])
    vibe, color  = CLUSTER_VIBES.get(top_cluster, (f"Cluster {top_cluster}", "#94a3b8"))

    a1, a2, a3 = st.columns(3)
    a1.metric("Most Common Cluster",  f"Cluster {top_cluster}")
    a2.metric("Vibe",                 vibe)
    a3.metric("Song Count",           len(artist_data))

    st.dataframe(
        artist_data[['name_song', 'cluster_label', 'release_date', 'popularity_songs']].head(10),
        use_container_width=True
    )

    fig_pie = px.pie(
        artist_data,
        names='cluster_label',
        title=f"Cluster Distribution for {selected_artist}",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0')
    st.plotly_chart(fig_pie, use_container_width=True)

#   FOOTER 
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#475569;'>Created by Farooque 🎓 | "
    "Amazon Music Clustering Project</div>",
    unsafe_allow_html=True
)
