import streamlit as st
import altair as alt
import polars as pl
import requests

st.set_page_config(page_title="Temperature Forecast", layout="wide")
st.title("🌡️ Temperature Forecast Visualization")


def load_data():
    try:
        response = requests.get("http://api-server:8000/forecast")

        # 요청 과도할 때 (429 응답)
        if response.status_code == 429:
            st.warning("🚨 요청이 너무 많습니다. 잠시 후 다시 시도해주세요.")
            return pl.DataFrame()

        if response.status_code == 200:
            data = response.json()
            data = pl.DataFrame(data)
            return data

        st.error(f"서버 오류: 상태코드 {response.status_code}")
        return pl.DataFrame()

    except Exception as e:
        st.error(f"📛 Error loading data: {e}")
        return pl.DataFrame()


def main():
    df = load_data()

    if df.is_empty():
        st.error("No data available!")
        return

    try:
        df = df.with_columns(
            pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.f")
        )
    except Exception as e:
        st.error(f"Error parsing datetime: {e}")
        try:
            df = df.with_columns(
                pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S")
            )
        except Exception as e2:
            st.error(f"Alternative datetime parsing also failed: {e2}")
            return

    if st.checkbox('Show raw data'):
        st.subheader('Raw Data')
        st.dataframe(df)

    st.subheader("Temperature Forecast")

    try:
        chart = df.plot.line(
            x=alt.X("datetime:T", title='Date and Time'),
            y=alt.Y("pred_temp:Q", title='Temperature (°C)',
                    scale=alt.Scale(domain=[15, 30])),
            tooltip=[
                alt.Tooltip("datetime:T", title='Date and Time'),
                alt.Tooltip("pred_temp:Q", title='Temperature (°C)')
            ]
        )

        st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating chart: {e}")
        st.write("Debug info:")
        st.write("DataFrame columns:", df.columns)
        st.write("DataFrame dtypes:", df.dtypes)
        if not df.is_empty():
            st.write("Sample data:", df.head())


def get_clothing_recommendations():
    try:
        response = requests.get("http://api-server:8000/clothing")

        if response.status_code == 429:
            st.warning("👕 추천 요청이 너무 많습니다. 잠시 후 다시 시도해주세요.")
            return []

        if response.status_code == 200:
            return response.json()

        st.error(f"서버 오류: 상태코드 {response.status_code}")
        return []

    except Exception as e:
        st.error(f"📛 Error loading clothing recommendations: {e}")
        return []


st.subheader("Clothing Recommendations")

recommendations = get_clothing_recommendations()
if recommendations:
    for rec in recommendations:
        with st.expander(f"📅 {rec['date']}"):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("🌡️ Avg Temp", f"{rec['avg_temp']:.1f}°C")
                st.metric("👕 Recommendation", rec['clothing'].split(' (')[0])
            with col2:
                st.write(f"🔹 Min: {rec['min_temp']:.1f}°C")
                st.write(f"🔹 Max: {rec['max_temp']:.1f}°C")
                st.caption(f"💡 {rec['clothing'].split('(')[1][:-1]}")
else:
    st.warning("No clothing recommendations available")


if __name__ == "__main__":
    main()
