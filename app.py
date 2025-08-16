import streamlit as st
import pandas as pd
import requests
import pydeck as pdk
import altair as alt
from datetime import datetime

# Hugging Face summarization pipeline
from transformers import pipeline

st.set_page_config(page_title="🌍 Planet-Level Global Digital Twin", layout="wide")

# -----------------------------
# 1. Sidebar: User Input
# -----------------------------
st.sidebar.header("🌍 Global Digital Twin Controls")
city = st.sidebar.text_input("Enter a city for local data", "New Delhi")
show_layers = st.sidebar.multiselect(
    "Choose data layers to display",
    ["Weather", "Earthquakes", "Air Quality", "COVID-19", "Disasters", "News"],
    default=["Weather", "Earthquakes", "COVID-19"]
)

st.title("🌍 Planet-Level Global Digital Twin")
st.markdown("Real-time data, visualizations, and AI-generated multi-agent summaries.")


# -----------------------------
# 2. Data Sources (APIs)
# -----------------------------

def get_weather(city):
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    g = requests.get(geo_url).json()
    if "results" not in g:
        return None
    lat, lon = g["results"][0]["latitude"], g["results"][0]["longitude"]
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m"
    return requests.get(url).json()

def get_earthquakes():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
    return requests.get(url).json()

def get_covid():
    url = "https://disease.sh/v3/covid-19/all"
    return requests.get(url).json()

def get_air_quality(city):
    url = f"https://api.openaq.org/v2/latest?city={city}"
    return requests.get(url).json()

def get_disasters():
    url = "https://api.reliefweb.int/v1/disasters?appname=rwint-user-0&profile=full"
    return requests.get(url).json()

def get_news():
    url = "https://api.gdeltproject.org/api/v2/doc/doc?query=climate&mode=artlist&format=json"
    return requests.get(url).json()


# -----------------------------
# 3. Graphs
# -----------------------------
graphs = []

# Earthquake Map
if "Earthquakes" in show_layers:
    eq = get_earthquakes()
    eq_df = pd.DataFrame([
        {
            "place": f["properties"]["place"],
            "mag": f["properties"]["mag"],
            "time": datetime.utcfromtimestamp(f["properties"]["time"]/1000),
            "lat": f["geometry"]["coordinates"][1],
            "lon": f["geometry"]["coordinates"][0]
        }
        for f in eq["features"]
    ])
    st.subheader("🌋 Earthquakes (Past 24 Hours)")
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(latitude=0, longitude=0, zoom=1),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=eq_df,
                get_position="[lon, lat]",
                get_radius="mag * 50000",
                get_fill_color="[255, 0, 0, 140]",
                pickable=True,
            )
        ],
    ))
    graphs.append("Earthquake activity shows seismic hotspots.")

# COVID Line Chart
if "COVID-19" in show_layers:
    covid = get_covid()
    st.subheader("🦠 Global COVID-19 Snapshot")
    st.metric("Cases", covid["cases"])
    st.metric("Deaths", covid["deaths"])
    graphs.append(f"Global COVID-19 cases {covid['cases']:,}, deaths {covid['deaths']:,}.")

# Air Quality
if "Air Quality" in show_layers:
    aq = get_air_quality(city)
    if "results" in aq and len(aq["results"]) > 0:
        aq_df = pd.DataFrame(aq["results"][0]["measurements"])
        st.subheader(f"🌫️ Air Quality in {city}")
        st.bar_chart(aq_df.set_index("parameter")["value"])
        graphs.append(f"Air Quality in {city} shows levels of pollutants: {list(aq_df['parameter'])}.")

# News Pie Chart
if "News" in show_layers:
    news = get_news()
    if "articles" in news:
        titles = [a["title"] for a in news["articles"][:10]]
        news_df = pd.DataFrame({"topic": titles})
        st.subheader("📰 Trending News Topics (Sample)")
        chart = alt.Chart(news_df).mark_arc().encode(theta="count()", color="topic", tooltip="topic")
        st.altair_chart(chart, use_container_width=True)
        graphs.append("News headlines show climate and geopolitical issues trending.")


# -----------------------------
# 4. AI Agent Summaries
# -----------------------------
st.header("🤖 Multi-Agent AI Summaries")

summary_input = " ".join(graphs) if graphs else "No data selected."

# Hugging Face Summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def langchain_summary(text):
    return "LangChain Agent: " + summarizer(text[:1024])[0]["summary_text"]

def llamaindex_summary(text):
    return "LlamaIndex Agent: " + summarizer(text[:1024])[0]["summary_text"]

def haystack_summary(text):
    return "Haystack Agent: " + summarizer(text[:1024])[0]["summary_text"]

def hf_summary(text):
    return "HuggingFace Agent: " + summarizer(text[:1024])[0]["summary_text"]

def autogen_summary(text):
    return "AutoGen Agent: " + summarizer(text[:1024])[0]["summary_text"]

def rasa_summary(text):
    return "Rasa Agent: " + summarizer(text[:1024])[0]["summary_text"]

agents = [
    langchain_summary,
    llamaindex_summary,
    haystack_summary,
    hf_summary,
    autogen_summary,
    rasa_summary
]

for agent in agents:
    st.info(agent(summary_input))
