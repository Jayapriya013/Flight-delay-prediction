import streamlit as st
from datetime import datetime, timedelta

# Streamlit UI
st.title("✈️ Flight Delay Checker (Rule-Based Logic)")

st.markdown("Check if a flight is delayed based on scheduled and actual departure times.")

with st.form("delay_form"):
    source = st.selectbox("Source Airport", ["JFK", "LAX", "ATL", "ORD", "DFW", "DEN"])
    destination = st.selectbox("Destination Airport", ["JFK", "LAX", "ATL", "ORD", "DFW", "DEN"])

    scheduled_time = st.time_input("Scheduled Departure Time (HH:MM)")
    actual_time = st.time_input("Actual Departure Time (HH:MM)")

    submitted = st.form_submit_button("Check Delay")

if submitted:
    if source == destination:
        st.warning("Source and destination airports must be different.")
    else:
        # Combine with today's date for datetime comparison
        today = datetime.today().date()
        scheduled_dt = datetime.combine(today, scheduled_time)
        actual_dt = datetime.combine(today, actual_time)

        # Handle flights crossing midnight
        if actual_dt < scheduled_dt:
            actual_dt += timedelta(days=1)

        # Calculate delay in minutes
        delay_minutes = (actual_dt - scheduled_dt).total_seconds() / 60

        if delay_minutes > 15:
            st.error(f"❌ Delayed by {int(delay_minutes)} minutes")
        else:
            st.success(f"✅ On Time (Delay: {int(delay_minutes)} minutes)")
