"""Evaluator UI (Laptop 2), Streamlit.
Run:  streamlit run frontend/app.py
Set the API address in the sidebar (or env var PATHSNAP_API), e.g. http://192.168.1.23:8000
"""
import os
import folium
import requests
import streamlit as st

st.set_page_config(page_title="PathSnap Evaluator", layout="wide")
st.title("PathSnap: Map-Matching Evaluation")

api = st.sidebar.text_input("API URL (Laptop 1)", os.environ.get("PATHSNAP_API", "http://127.0.0.1:8000")).rstrip("/")
st.sidebar.caption("Change this to Laptop 1's IP on the day.")


@st.cache_data(ttl=10, show_spinner=False)
def get_trips(url):
    r = requests.get(f"{url}/trips", timeout=10)
    r.raise_for_status()
    return r.json()["trips"]


def build_map(points, segments):
    m = folium.Map(location=points[0], zoom_start=15, tiles="OpenStreetMap")
    gps = folium.FeatureGroup(name="Raw GPS points (blue)")
    for p in points:
        folium.CircleMarker(p, radius=2, color="#2563eb", fill=True, fill_opacity=0.8).add_to(gps)
    gps.add_to(m)
    pred = folium.FeatureGroup(name="Predicted route (red)")
    for s in segments:
        folium.PolyLine(s, color="#dc2626", weight=4, opacity=0.8).add_to(pred)
    pred.add_to(m)
    lats, lons = [p[0] for p in points], [p[1] for p in points]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    folium.LayerControl().add_to(m)
    return m


try:
    trips = get_trips(api)
except Exception as e:
    st.error(f"Cannot reach the API at {api}. Is Laptop 1 running uvicorn with --host 0.0.0.0?\n\n{e}")
    st.stop()

st.subheader("Select an evaluation trip")
choice = st.radio("Trips", trips, horizontal=True, label_visibility="collapsed")

if st.button("Run prediction", type="primary"):
    with st.spinner("Running the pipeline on Laptop 1..."):
        try:
            r = requests.post(f"{api}/predict", json={"trip_id": choice}, timeout=120)
            r.raise_for_status()
            st.session_state["result"] = r.json()
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(f"Prediction failed: {e}")

res = st.session_state.get("result")
if res:
    m = res["metrics"]
    st.success(f"Trip {res['trip_id']} done in {res['latency_ms']} ms")
    c = st.columns(4)
    c[0].metric("GPS points", m["n_points"])
    c[1].metric("Accuracy", f"{m['accuracy']*100:.1f}%")
    c[2].metric("Route continuity", f"{m['route_continuity']*100:.1f}%")
    c[3].metric("Unique predicted segments", m["n_unique_predicted_arcs"])
    map_html = build_map(res["points"], res["segments"])._repr_html_()
    if hasattr(st, "iframe"):                      # newer Streamlit
        st.iframe(map_html, height=620)
    else:                                          # older Streamlit
        import streamlit.components.v1 as components
        components.html(map_html, height=620)
