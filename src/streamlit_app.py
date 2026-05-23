"""Streamlit interface for the marketplace quality scorer."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

from src.inference import ProductInput, dump_product, estimate_quality


DEFAULT_API_URL = "http://localhost:8000/predict"


st.set_page_config(
    page_title="Marketplace Quality Studio",
    page_icon="📦",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7f3ec 0%, #fffaf4 40%, #f2efe9 100%);
    }
    .hero-card {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(20, 20, 20, 0.08);
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.08);
    }
    .section-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 18px;
        padding: 18px;
        border: 1px solid rgba(20, 20, 20, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _submit_to_api(payload: dict[str, object], api_url: str) -> dict[str, object]:
    request = Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=20) as response:  # nosec B310 - local API endpoint
        return json.loads(response.read().decode("utf-8"))


def _product_form() -> ProductInput:
    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    st.title("Marketplace Quality Studio")
    st.caption("Веб-интерфейс для оценки карточки товара через Streamlit и FastAPI.")

    with st.form("quality_form"):
        col_left, col_right = st.columns(2)

        with col_left:
            article = st.text_input("Артикул", value="WB-12345")
            name = st.text_input("Название", value="Кроссовки мужские демисезонные на каждый день")
            description = st.text_area(
                "Описание",
                value="Легкая модель из износостойких материалов с амортизирующей подошвой.",
                height=120,
            )
            brand = st.text_input("Бренд", value="HSEGenius")
            category = st.text_input("Категория", value="Мужские кеды/кроссовки")
            price = st.number_input("Цена", min_value=0.0, value=4990.0, step=100.0)
            special_price = st.number_input("Спец. цена", min_value=0.0, value=3990.0, step=100.0)
            old_price = st.number_input("Старая цена", min_value=0.0, value=6990.0, step=100.0)
            discount = st.number_input("Скидка", min_value=0.0, value=43.0, step=1.0)
            images_count = st.slider("Кол-во картинок", min_value=0, max_value=20, value=8)

        with col_right:
            color = st.text_input("Цвет", value="Черный")
            size = st.text_input("Размер", value="42")
            seller = st.text_input("Продавец", value="Demo Seller")
            fulfillment = st.text_input("Фулфилмент", value="FBO")
            delivery = st.text_input("Доставка", value="WB")
            season = st.text_input("Сезон", value="Демисезон")
            size_name = st.text_input("Название размера", value="RU 42")
            rich_content = st.selectbox("Rich-контент", ["Да", "Нет"], index=0)
            country = st.text_input("Страна производства", value="Китай")
            video_cover = st.selectbox("Видеообложка", ["Да", "Нет"], index=0)
            advertising = st.selectbox("Реклама", ["Нет", "Да"], index=0)
            rating = st.slider("Рейтинг", min_value=1.0, max_value=5.0, value=4.7, step=0.1)
            reviews = st.number_input("Оценок", min_value=0.0, value=126.0, step=1.0)

        submitted = st.form_submit_button("Оценить карточку")

    st.markdown("</div>", unsafe_allow_html=True)

    if not submitted:
        st.stop()

    return ProductInput(
        **{
            "Артикул": article,
            "Название": name,
            "Описание": description,
            "Бренд": brand,
            "Категория": category,
            "Цена": price,
            "Спец. цена": special_price,
            "Старая цена": old_price,
            "Скидка": discount,
            "Цвет": color,
            "Размер": size,
            "Продавец": seller,
            "Фулфилмент": fulfillment,
            "Доставка": delivery,
            "Сезон": season,
            "Название размера": size_name,
            "Кол-во картинок": images_count,
            "Rich-контент": rich_content,
            "Страна производства": country,
            "Видеообложка": video_cover,
            "Реклама": advertising,
            "Рейтинг": rating,
            "Оценок": reviews,
        }
    )


def main() -> None:
    api_url = st.sidebar.text_input("FastAPI endpoint", value=DEFAULT_API_URL)
    st.sidebar.caption("Если API не запущен, приложение использует локальный baseline-сервис.")

    product = _product_form()
    payload = dump_product(product)

    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Запрос")
        st.json(payload)
        st.markdown("</div>", unsafe_allow_html=True)

    try:
        result = _submit_to_api(payload, api_url)
        source = "FastAPI"
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        prediction = estimate_quality(product)
        result = prediction.model_dump() if hasattr(prediction, "model_dump") else prediction.dict()
        source = "local baseline"

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Результат")
        st.metric("Score", f'{result["score_percent"]}%')
        st.progress(int(result["score_percent"]))
        st.write(f'Класс: **{result["grade"]}**')
        st.write(f'Источник расчёта: {source}')
        st.write(result["explanation"])
        st.markdown("#### Сигналы")
        st.json(result["signals"])
        st.markdown("#### Что улучшить")
        for item in result["recommendations"]:
            st.write(f"- {item}")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
