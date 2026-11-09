"""
app.py
======
AI Travel Planner Agent — Streamlit UI
Powered by Llama 3.3 on IBM watsonx.ai — IBM Cloud Lite tier compatible
"""

import os
import streamlit as st
from dotenv import load_dotenv

from agent_tools import (
    get_weather,
    get_destinations,
    build_itinerary_prompt,
    get_transport_options,
    get_accommodation_options,
)
from llm_agent import generate_text, chat_with_agent, get_active_model_id, get_model_label

# Resolved once per app run so every reference stays consistent.
_model_id    = get_active_model_id()
_model_label = get_model_label()

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="✈️ AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS  (base theme colours are set in .streamlit/config.toml)
# ---------------------------------------------------------------------------
# Palette reference (WCAG AA compliant against their backgrounds):
#   --bg:          #0F1B24   main app background
#   --sidebar-bg:  #16252F   sidebar background
#   --card:        #1F3342   card / surface
#   --accent:      #4DA3FF   primary accent / links
#   --text:        #F1F5F9   primary text  (≥ 7:1 on --bg)
#   --text-muted:  #B6C2CE   secondary text (≥ 4.6:1 on --bg)
#   --input-bg:    #223645   input background
#   --border:      #35505F   border colour

st.markdown("""
<style>
    /* ── Typography ─────────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", sans-serif !important;
        font-size: 16px;
        color: #F1F5F9;
    }

    /* ── App & sidebar backgrounds ──────────────────────────────────────── */
    .stApp                            { background: #0F1B24 !important; }
    [data-testid="stSidebar"]         { background: #16252F !important; }
    [data-testid="stSidebar"] > div   { background: #16252F !important; }
    .main .block-container            { padding-top: 1.5rem; }

    /* ── Headings ────────────────────────────────────────────────────────── */
    h1, h2, h3, h4 { color: #F1F5F9 !important; font-weight: 700; }
    h1 { font-size: 2rem; }
    h2 { font-size: 1.5rem; }
    h3 { font-size: 1.25rem; }

    /* Sidebar section title */
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #F1F5F9 !important; }

    /* ── Body text & labels ──────────────────────────────────────────────── */
    p, li, span, td, th           { color: #F1F5F9; }
    label, .stWidgetLabel > label  { color: #B6C2CE !important; font-size: 14px !important; }
    .stCaption, .stCaption p       { color: #B6C2CE !important; font-size: 13px; }

    /* Markdown paragraphs inside sidebar */
    [data-testid="stSidebar"] p    { color: #F1F5F9 !important; }

    /* ── Inputs: text_input, number_input, text_area, selectbox ─────────── */
    /* text / number input box */
    .stTextInput  input,
    .stNumberInput input {
        background: #223645 !important;
        color: #FFFFFF    !important;
        border: 1px solid #35505F !important;
        border-radius: 6px !important;
        font-size: 15px !important;
    }
    .stTextInput  input::placeholder,
    .stNumberInput input::placeholder { color: #7A9AB5 !important; }

    /* text_area */
    .stTextArea textarea {
        background: #223645 !important;
        color: #FFFFFF    !important;
        border: 1px solid #35505F !important;
        border-radius: 6px !important;
        font-size: 15px !important;
    }
    .stTextArea textarea::placeholder { color: #7A9AB5 !important; }

    /* selectbox trigger */
    .stSelectbox [data-baseweb="select"] > div:first-child {
        background: #223645 !important;
        border: 1px solid #35505F !important;
        border-radius: 6px !important;
    }
    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] div { color: #FFFFFF !important; }

    /* selectbox dropdown list */
    [data-baseweb="popover"] ul,
    [data-baseweb="menu"]    li {
        background: #1F3342 !important;
        color: #F1F5F9 !important;
    }
    [data-baseweb="menu"] li:hover { background: #2A4A5E !important; }

    /* number_input +/- stepper buttons */
    .stNumberInput button {
        background: #2A4A5E !important;
        color: #F1F5F9    !important;
        border: 1px solid #35505F !important;
    }
    .stNumberInput button:hover { background: #35505F !important; }

    /* ── Tabs ────────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"]                { background: transparent; border-bottom: 1px solid #35505F; }
    .stTabs [data-baseweb="tab"]                     { color: #B6C2CE; font-size: 15px; font-weight: 500; background: transparent; }
    .stTabs [data-baseweb="tab"]:hover               { color: #F1F5F9; }
    .stTabs [data-baseweb="tab"][aria-selected="true"]{ color: #FFFFFF; border-bottom: 2px solid #4DA3FF; font-weight: 700; }

    /* ── Buttons ─────────────────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(90deg, #1565c0, #0d47a1);
        color: #FFFFFF !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 15px;
        padding: 0.45rem 1rem;
        transition: background 0.2s ease;
    }
    .stButton > button:hover  { background: linear-gradient(90deg, #1976d2, #1565c0) !important; color: #FFFFFF !important; }
    .stButton > button:active { background: #0d47a1 !important; }

    /* Form submit button */
    .stFormSubmitButton > button {
        background: linear-gradient(90deg, #1565c0, #0d47a1);
        color: #FFFFFF !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 15px;
    }
    .stFormSubmitButton > button:hover { background: linear-gradient(90deg, #1976d2, #1565c0) !important; }

    /* ── Alert / info boxes ──────────────────────────────────────────────── */
    .stInfo, [data-testid="stInfo"]            { background: #1A3350 !important; color: #C8DEFF !important; border-left: 4px solid #4DA3FF; border-radius: 6px; }
    .stSuccess, [data-testid="stSuccess"]      { background: #14362A !important; color: #A7F3D0 !important; border-left: 4px solid #34D399; border-radius: 6px; }
    .stWarning, [data-testid="stWarning"]      { background: #3B2F0F !important; color: #FDE68A !important; border-left: 4px solid #FBBF24; border-radius: 6px; }
    .stError, [data-testid="stError"]          { background: #3B1010 !important; color: #FCA5A5 !important; border-left: 4px solid #EF4444; border-radius: 6px; }
    /* text inside alert boxes */
    .stInfo p, .stSuccess p, .stWarning p, .stError p { color: inherit !important; }

    /* ── Cards ───────────────────────────────────────────────────────────── */
    .travel-card {
        background: #1F3342;
        border: 1px solid #35505F;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        color: #F1F5F9;
    }
    .travel-card h4            { color: #4DA3FF !important; margin-bottom: 0.4rem; font-size: 1rem; font-weight: 700; }
    .travel-card p, .travel-card li, .travel-card td, .travel-card span { color: #F1F5F9 !important; }

    /* ── Weather widget ──────────────────────────────────────────────────── */
    .weather-big { font-size: 3rem; font-weight: 700; color: #FFD54F; line-height: 1; }

    /* ── IBM badge ───────────────────────────────────────────────────────── */
    .ibm-badge {
        display: inline-block;
        background: #0062ff;
        color: #FFFFFF;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        margin-left: 6px;
        vertical-align: middle;
    }

    /* ── Chat bubbles ────────────────────────────────────────────────────── */
    .chat-user {
        background: #1565c0;
        border-radius: 12px 12px 4px 12px;
        padding: 0.7rem 1rem;
        margin: 0.5rem 0;
        color: #E3F2FD !important;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-assistant {
        background: #1F3342;
        border: 1px solid #35505F;
        border-radius: 12px 12px 12px 4px;
        padding: 0.7rem 1rem;
        margin: 0.5rem 0;
        color: #F1F5F9 !important;
        max-width: 85%;
    }

    /* ── Download button ─────────────────────────────────────────────────── */
    .stDownloadButton > button {
        background: #1F3342 !important;
        color: #4DA3FF   !important;
        border: 1px solid #4DA3FF !important;
        border-radius: 8px;
        font-weight: 600;
    }
    .stDownloadButton > button:hover { background: #2A4A5E !important; }

    /* ── Horizontal rule ─────────────────────────────────────────────────── */
    hr { border-color: #35505F !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "itinerary" not in st.session_state:
    st.session_state.itinerary = ""
if "last_weather" not in st.session_state:
    st.session_state.last_weather = None


# ---------------------------------------------------------------------------
# Sidebar — User Preferences
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ✈️ Travel Preferences")
    st.markdown("---")

    origin_city = st.text_input("🏠 Your Origin City", value="Mumbai", key="origin")
    destination_city = st.text_input("📍 Destination City / Country", value="Paris", key="dest")

    col1, col2 = st.columns(2)
    with col1:
        trip_days = st.number_input("📅 Trip Duration (days)", min_value=1, max_value=30, value=7)
    with col2:
        num_travelers = st.number_input("👥 Travelers", min_value=1, max_value=20, value=2)

    travel_style = st.selectbox(
        "🎭 Travel Style",
        ["Cultural Explorer", "Beach & Relaxation", "Adventure & Sports",
         "Luxury", "Budget Backpacker", "Family Friendly", "Romantic Getaway"],
    )

    budget_level = st.selectbox(
        "💰 Budget Level",
        ["Low (< $50/day)", "Medium ($50–$150/day)", "High ($150–$300/day)", "Very High (> $300/day)"],
    )

    interests = st.text_area(
        "🎯 Interests & Must-Sees",
        value="local food, museums, photography, walking tours",
        height=80,
    )

    st.markdown("---")
    st.markdown("### 🔧 AI Model Status")
    ibm_key = os.getenv("IBM_API_KEY", "")
    if ibm_key and ibm_key != "your_ibm_api_key_here":
        st.success("✅ IBM watsonx.ai connected")
    else:
        st.warning("⚠️ Demo mode — Add IBM API key in .env to enable AI generation")
    st.markdown(f'<span class="ibm-badge">{_model_label}</span> `{_model_id.split("/")[-1]}`', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Powered by IBM watsonx.ai · IBM Cloud Lite")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(f"""
<div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
    <h1>✈️ AI Travel Planner Agent</h1>
    <p style='color:#90caf9; font-size:1.05rem;'>
        Powered by <span style='color:#64b5f6; font-weight:700;'>{_model_label}</span> on
        <span style='color:#ffd54f; font-weight:700;'>watsonx.ai</span> · IBM Cloud Lite
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Itinerary Planner",
    "🌤️ Weather & Destinations",
    "🚌 Transport & Stay",
    "💬 AI Chat Assistant",
    "📋 Trip Summary",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Itinerary Planner
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### 🗺️ AI-Powered Itinerary Generator")
    st.markdown(
        f"Plan your **{trip_days}-day** trip to **{destination_city}** with a personalized day-by-day schedule."
    )

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown('<div class="travel-card">', unsafe_allow_html=True)
        st.markdown("#### 📝 Trip Summary")
        st.markdown(f"**From:** {origin_city}")
        st.markdown(f"**To:** {destination_city}")
        st.markdown(f"**Duration:** {trip_days} days")
        st.markdown(f"**Travelers:** {num_travelers}")
        st.markdown(f"**Style:** {travel_style}")
        st.markdown(f"**Budget:** {budget_level}")
        st.markdown(f"**Interests:** {interests}")
        st.markdown("</div>", unsafe_allow_html=True)

        generate_btn = st.button("🚀 Generate Itinerary with AI", use_container_width=True)

    with col_right:
        if generate_btn:
            with st.spinner(f"🤖 {_model_label} is crafting your perfect itinerary..."):
                prompt = build_itinerary_prompt(
                    destination=destination_city,
                    days=trip_days,
                    travel_style=travel_style,
                    budget=budget_level,
                    interests=interests,
                )
                result = generate_text(prompt)
                st.session_state.itinerary = result

        if st.session_state.itinerary:
            st.markdown("#### 📅 Your Personalized Itinerary")
            st.markdown(
                f'<div class="travel-card">{st.session_state.itinerary.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Generated by `{_model_id}`")
            st.download_button(
                label="📥 Download Itinerary",
                data=st.session_state.itinerary,
                file_name=f"itinerary_{destination_city.replace(' ','_')}_{trip_days}days.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.info("👈 Configure your preferences in the sidebar, then click **Generate Itinerary** to get started.")
            st.markdown("""
<div class="travel-card">
<h4>🌟 What the AI will generate for you:</h4>
<ul>
  <li>Day-by-day morning, afternoon & evening activities</li>
  <li>Specific landmark & restaurant recommendations</li>
  <li>Local transport tips for each day</li>
  <li>Estimated daily spend aligned to your budget</li>
  <li>Destination-specific packing checklist</li>
</ul>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Weather & Destinations
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### 🌤️ Weather & Destination Explorer")

    col_w1, col_w2 = st.columns(2)

    with col_w1:
        st.markdown("#### Current Weather")
        weather_city = st.text_input("Search city weather", value=destination_city, key="weather_city_input")
        if st.button("🔍 Fetch Weather", use_container_width=True):
            with st.spinner(f"Fetching weather for {weather_city}..."):
                weather = get_weather(weather_city)
                st.session_state.last_weather = weather

        if st.session_state.last_weather:
            w = st.session_state.last_weather
            if "error" in w:
                st.error(w["error"])
            else:
                if w.get("mock"):
                    st.caption("📌 Demo data — add OPENWEATHER_API_KEY in .env for live weather")

                st.markdown(f"""
<div class="travel-card">
  <h4>🌍 {w['city']}, {w['country']}</h4>
  <div class="weather-big">{w['temperature_c']}°C</div>
  <p style='color:#90caf9;font-size:1.1rem;'>{w['description'].title()}</p>
  <table style='color:#cfd8dc;width:100%;'>
    <tr><td>🌡️ Feels like</td><td><b>{w['feels_like_c']}°C</b></td></tr>
    <tr><td>💧 Humidity</td><td><b>{w['humidity_pct']}%</b></td></tr>
    <tr><td>🌬️ Wind</td><td><b>{w['wind_kmh']} km/h</b></td></tr>
  </table>
</div>
""", unsafe_allow_html=True)

                st.markdown("**📆 5-Day Forecast**")
                forecast_cols = st.columns(min(5, len(w["forecast"])))
                for i, day in enumerate(w["forecast"][:5]):
                    with forecast_cols[i]:
                        date_str = day["date"][:10]
                        st.markdown(f"""
<div class="travel-card" style='text-align:center;padding:0.7rem;'>
  <div style='font-size:0.8rem;color:#90caf9;'>{date_str}</div>
  <div style='font-size:1.4rem;font-weight:700;color:#ffd54f;'>{day['temp_c']}°C</div>
  <div style='font-size:0.75rem;color:#b0bec5;'>{day['description']}</div>
</div>""", unsafe_allow_html=True)

    with col_w2:
        st.markdown("#### 🗺️ Destination Recommendations")
        dest_type = st.selectbox(
            "I want to visit a...",
            ["Beach", "Mountain", "Cultural", "Adventure", "City"],
            key="dest_type_select",
        )
        budget_filter = st.selectbox(
            "Budget filter (optional)",
            ["All", "Low", "Medium", "High", "Very High"],
            key="budget_filter_select",
        )

        if st.button("✨ Find Destinations", use_container_width=True):
            budget_arg = None if budget_filter == "All" else budget_filter
            destinations = get_destinations(dest_type.lower(), budget_arg)

            st.markdown(f"**Top {dest_type} Destinations:**")
            for d in destinations:
                st.markdown(f"""
<div class="travel-card">
  <h4>📍 {d['name']}</h4>
  <p style='color:#cfd8dc;margin:0;'>🎯 {d['highlights']}</p>
  <p style='color:#b0bec5;margin:0.3rem 0 0 0;'>
    💰 Budget: <b style='color:#ffd54f;'>{d['budget']}</b> &nbsp;|&nbsp;
    📅 Best time: <b style='color:#a5d6a7;'>{d['best_months']}</b>
  </p>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Transport & Accommodation
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### 🚌 Transport & Accommodation Options")

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("#### 🛫 Transport Options")
        t_origin = st.text_input("From", value=origin_city, key="t_origin")
        t_dest = st.text_input("To", value=destination_city, key="t_dest")
        if st.button("🔎 Show Transport Options", use_container_width=True):
            transport = get_transport_options(t_origin, t_dest)
            st.markdown(f"**Options from {transport['origin']} → {transport['destination']}**")
            for opt in transport["options"]:
                st.markdown(f"""
<div class="travel-card">
  <h4>{opt['mode']}</h4>
  <p style='color:#cfd8dc;margin:0;'>⏱ Duration: {opt['typical_duration']}</p>
  <p style='color:#cfd8dc;margin:0;'>💵 Cost: <b style='color:#ffd54f;'>${opt['cost_range_usd']}</b></p>
  <p style='color:#90caf9;margin:0.3rem 0 0 0;font-size:0.9rem;'>💡 {opt['tip']}</p>
</div>""", unsafe_allow_html=True)

    with col_t2:
        st.markdown("#### 🏨 Accommodation Options")
        acc_dest = st.text_input("Destination for stay", value=destination_city, key="acc_dest")
        acc_budget = st.selectbox(
            "Your budget tier",
            ["Low", "Medium", "High", "Very High"],
            key="acc_budget",
        )
        if st.button("🔎 Show Accommodation Options", use_container_width=True):
            accommodations = get_accommodation_options(acc_dest, acc_budget)
            st.markdown(f"**Accommodation options in {acc_dest}:**")
            for acc in accommodations:
                st.markdown(f"""
<div class="travel-card">
  <h4>🛏️ {acc['type']}</h4>
  <p style='color:#cfd8dc;margin:0;'>💵 <b style='color:#ffd54f;'>${acc['cost_usd_night']}</b> / night</p>
  <p style='color:#90caf9;margin:0.3rem 0 0 0;font-size:0.9rem;'>🔗 Platforms: {acc['platforms']}</p>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 💡 Booking Tips")
        st.markdown("""
<div class="travel-card">
  <ul style='color:#cfd8dc;margin:0;'>
    <li>Book flights <b style='color:#ffd54f;'>6–8 weeks ahead</b> for best prices</li>
    <li>Use <b style='color:#64b5f6;'>Google Flights</b> price alerts for fare drops</li>
    <li>Check <b style='color:#64b5f6;'>Booking.com Free Cancellation</b> for flexibility</li>
    <li>Hostels often include free walking tours — great for solo travelers</li>
    <li>Airbnb monthly discounts can be 30–50% cheaper for long stays</li>
    <li>Always buy <b style='color:#ffd54f;'>travel insurance</b> before departure</li>
  </ul>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — AI Chat Assistant
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("### 💬 Chat with Your AI Travel Planner")
    st.markdown(
        "Ask anything about your trip — visa requirements, local customs, "
        "hidden gems, safety tips, food recommendations, and more."
    )

    # Display conversation history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(f"""
<div class="travel-card">
  <h4>👋 Hello! I'm your AI Travel Planner powered by {_model_label}.</h4>
  <p style='color:#cfd8dc;'>I can help you with:</p>
  <ul style='color:#b0bec5;'>
    <li>🌍 Destination advice and hidden gems</li>
    <li>📋 Visa & entry requirements</li>
    <li>🍜 Local food & restaurant recommendations</li>
    <li>🛡️ Safety tips & travel advisories</li>
    <li>💱 Currency, tipping culture & local etiquette</li>
    <li>📦 Packing lists tailored to your destination</li>
  </ul>
  <p style='color:#90caf9;'>Type a message below to get started!</p>
</div>
""", unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="chat-user">👤 {msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="chat-assistant">🤖 <b>{_model_label}:</b><br>{msg["content"].replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown("---")

    # Suggested questions
    st.markdown("**💡 Quick questions:**")
    quick_cols = st.columns(3)
    quick_questions = [
        f"What visa do I need for {destination_city}?",
        f"Best local foods to try in {destination_city}",
        f"Is {destination_city} safe for solo travelers?",
        f"What currency is used in {destination_city}?",
        f"Top hidden gems in {destination_city}",
        f"What to pack for {trip_days} days in {destination_city}?",
    ]
    for i, qcol in enumerate(quick_cols):
        for j in range(2):
            idx = i + j * 3
            if idx < len(quick_questions):
                if qcol.button(quick_questions[idx], key=f"quick_{idx}", use_container_width=True):
                    st.session_state._quick_q = quick_questions[idx]

    # Check if a quick question was clicked
    auto_question = st.session_state.pop("_quick_q", None)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your message",
            value=auto_question or "",
            placeholder="Ask about your trip to " + destination_city + "...",
            label_visibility="collapsed",
        )
        send_btn = st.form_submit_button("Send 📨", use_container_width=True)

    if send_btn and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        with st.spinner(f"🤖 {_model_label} is thinking..."):
            reply = chat_with_agent(user_input.strip(), st.session_state.chat_history[:-1])
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — Trip Summary
# ═══════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("### 📋 Trip Overview & Smart Checklist")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("""
<div class="travel-card">
<h4>📌 Trip Details</h4>
""" + f"""
<table style='color:#cfd8dc;width:100%;border-collapse:collapse;'>
  <tr><td style='padding:4px 0;'>🏠 Origin</td><td><b>{origin_city}</b></td></tr>
  <tr><td style='padding:4px 0;'>📍 Destination</td><td><b>{destination_city}</b></td></tr>
  <tr><td style='padding:4px 0;'>📅 Duration</td><td><b>{trip_days} days</b></td></tr>
  <tr><td style='padding:4px 0;'>👥 Travelers</td><td><b>{num_travelers}</b></td></tr>
  <tr><td style='padding:4px 0;'>🎭 Style</td><td><b>{travel_style}</b></td></tr>
  <tr><td style='padding:4px 0;'>💰 Budget</td><td><b>{budget_level}</b></td></tr>
  <tr><td style='padding:4px 0;'>🎯 Interests</td><td><b>{interests}</b></td></tr>
</table>
</div>
""", unsafe_allow_html=True)

        # Smart checklist
        st.markdown("""
<div class="travel-card">
<h4>✅ Pre-Travel Checklist</h4>
<ul style='color:#cfd8dc;list-style:none;padding-left:0;'>
  <li>🛂 Check passport validity (6+ months)</li>
  <li>📋 Research & apply for visa if required</li>
  <li>✈️ Book flights (6–8 weeks ahead)</li>
  <li>🏨 Reserve accommodation</li>
  <li>🛡️ Purchase travel insurance</li>
  <li>💉 Check vaccination requirements</li>
  <li>💱 Notify bank of travel plans</li>
  <li>📱 Download offline maps (Google Maps)</li>
  <li>🔌 Pack universal power adapter</li>
  <li>📞 Get local SIM or international plan</li>
  <li>🔔 Register with your country's embassy</li>
  <li>📸 Make digital copies of documents</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with col_s2:
        st.markdown("#### 🤖 Get AI Trip Summary")
        if st.button("📊 Generate Smart Trip Summary with AI", use_container_width=True):
            summary_prompt = f"""Provide a comprehensive travel summary for:
- Trip: {origin_city} to {destination_city}
- Duration: {trip_days} days, {num_travelers} traveler(s)
- Style: {travel_style}, Budget: {budget_level}
- Interests: {interests}

Include:
1. Why {destination_city} is a great choice for this traveler
2. Top 5 must-do experiences
3. Cultural etiquette & local tips
4. Estimated total trip budget breakdown (flights, hotel, food, activities)
5. Best time to visit & current season info
6. One unique local experience most tourists miss

Be concise but specific."""
            with st.spinner("🤖 Generating your personalized trip summary..."):
                summary = generate_text(summary_prompt)
            st.markdown(
                f'<div class="travel-card">{summary.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )
            if st.session_state.itinerary:
                st.download_button(
                    label="📥 Download Full Trip Plan",
                    data=f"TRIP SUMMARY\n{'='*50}\n{summary}\n\nITINERARY\n{'='*50}\n{st.session_state.itinerary}",
                    file_name=f"trip_plan_{destination_city.replace(' ','_')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
        else:
            st.markdown("""
<div class="travel-card">
  <h4>🌟 Smart Trip Summary includes:</h4>
  <ul style='color:#cfd8dc;'>
    <li>Why this destination suits you</li>
    <li>Top 5 must-do experiences</li>
    <li>Cultural etiquette & local tips</li>
    <li>Full budget breakdown</li>
    <li>Best season to visit</li>
    <li>Hidden local gem recommendation</li>
  </ul>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="travel-card">
<h4>🔗 Useful Travel Resources</h4>
<ul style='color:#b0bec5;'>
  <li><a href='https://www.skyscanner.com' style='color:#64b5f6;'>Skyscanner</a> — Compare flight prices</li>
  <li><a href='https://www.booking.com' style='color:#64b5f6;'>Booking.com</a> — Hotel reservations</li>
  <li><a href='https://www.airbnb.com' style='color:#64b5f6;'>Airbnb</a> — Unique accommodations</li>
  <li><a href='https://www.tripadvisor.com' style='color:#64b5f6;'>TripAdvisor</a> — Reviews & guides</li>
  <li><a href='https://www.xe.com' style='color:#64b5f6;'>XE Currency</a> — Live exchange rates</li>
  <li><a href='https://www.iatatravelcentre.com' style='color:#64b5f6;'>IATA Travel Centre</a> — Visa requirements</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(f"""
<div style='text-align:center;padding:1rem 0;color:#546e7a;font-size:0.85rem;'>
  ✈️ <b>AI Travel Planner Agent</b> &nbsp;·&nbsp;
  Powered by <span style='color:#0062ff;font-weight:700;'>{_model_label}</span> on
  <span style='color:#0062ff;'>watsonx.ai</span> &nbsp;·&nbsp; IBM Cloud Lite &nbsp;·&nbsp;
  Built with <a href='https://streamlit.io' style='color:#64b5f6;'>Streamlit</a>
</div>
""", unsafe_allow_html=True)
