import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.title("Vancouver Neighbourhood Business Composition")

@st.cache_data
def load_data():
    df = pd.read_parquet("data/businesses_clean.parquet")
    counts = df["localarea"].value_counts()
    keep = counts[counts >= 100].index
    sub = df[df["localarea"].isin(keep)]

    comp = pd.crosstab(sub["localarea"], sub["businesstype_grouped"],
                       normalize="index") * 100
    centroids = sub.groupby("localarea")[["geo_point_2d.lon",
                                          "geo_point_2d.lat"]].mean()
    return comp, centroids, counts[keep]

comp, centroids, counts = load_data()

# PCA fit once, outside the K logic
coords_pca = PCA(n_components=2).fit_transform(comp.values)

k = st.sidebar.slider("Number of clusters", 2, 8, 4)
labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(comp.values)

order = {"cluster": [str(i) for i in range(k)]}

st.sidebar.caption(
    "Cluster numbers and colours are assigned fresh each time K changes. "
    "Cluster 2 at K=4 has no relationship to cluster 2 at K=5."
)

# B2 - PCA scatter of areas coloured by cluster
st.subheader("Area similarity (PCA projection)")

plot_df = pd.DataFrame({
    "PC1": coords_pca[:, 0],
    "PC2": coords_pca[:, 1],
    "area": comp.index,
    "cluster": labels.astype(str),
})

fig = px.scatter(
    plot_df, x="PC1", y="PC2", color="cluster",
    hover_name="area", text="area",
    category_orders=order, height=600,
)
fig.update_traces(marker=dict(size=12), textposition="top center")
st.plotly_chart(fig, use_container_width=True)

# B3 - geographic view of area centroids
st.subheader("Geographic view")

map_df = pd.DataFrame({
    "area": comp.index,
    "lat": centroids.loc[comp.index, "geo_point_2d.lat"].values,
    "lon": centroids.loc[comp.index, "geo_point_2d.lon"].values,
    "businesses": counts.loc[comp.index].values,
    "cluster": labels.astype(str),
    "size_scaled": counts.loc[comp.index].values ** 0.5,
})

map_fig = px.scatter_map(
    map_df,
    lat="lat", lon="lon",
    color="cluster",
    size="size_scaled",
    hover_name="area",
    hover_data={"businesses": True, "lat": False, "lon": False},
    category_orders=order,
    zoom=11,
    height=600,
    map_style="carto-positron",
    size_max=40,
)
map_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(map_fig, use_container_width=True)

# B4 - cluster membership and profiling
st.subheader("Cluster membership")

members = pd.DataFrame({"area": comp.index, "cluster": labels})
overall = comp.mean()

for c in sorted(members["cluster"].unique()):
    areas = members[members["cluster"] == c]["area"].tolist()
    diff = (comp.loc[areas].mean() - overall).sort_values(ascending=False)
    top = ", ".join(f"{name} ({val:+.1f})" for name, val in diff.head(3).items())

    st.markdown(f"**Cluster {c}** ({len(areas)} areas)")
    st.markdown(f"{', '.join(areas)}")
    st.caption(f"Most over-represented: {top}")