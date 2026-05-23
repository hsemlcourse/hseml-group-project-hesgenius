from fastapi.testclient import TestClient

from src.api import app
from src.inference import ProductInput, estimate_quality


def _sample_product() -> ProductInput:
    return ProductInput(
        **{
            "Артикул": "WB-12345",
            "Название": "Кроссовки мужские демисезонные на каждый день",
            "Описание": "Легкая модель из износостойких материалов с амортизирующей подошвой.",
            "Бренд": "HSEGenius",
            "Категория": "Мужские кеды/кроссовки",
            "Цена": 4990,
            "Спец. цена": 3990,
            "Старая цена": 6990,
            "Скидка": 43,
            "Цвет": "Черный",
            "Размер": "42",
            "Продавец": "Demo Seller",
            "Фулфилмент": "FBO",
            "Доставка": "WB",
            "Сезон": "Демисезон",
            "Название размера": "RU 42",
            "Кол-во картинок": 8,
            "Rich-контент": "Да",
            "Страна производства": "Китай",
            "Видеообложка": "Да",
            "Реклама": "Нет",
            "Рейтинг": 4.7,
            "Оценок": 126,
        }
    )


def test_estimate_quality_returns_bounded_score() -> None:
    prediction = estimate_quality(_sample_product())

    assert 0.0 <= prediction.score <= 1.0
    assert prediction.score_percent == int(round(prediction.score * 100))
    assert prediction.grade in {"Высокое качество", "Среднее качество", "Нужна доработка"}


def test_predict_endpoint_returns_json() -> None:
    client = TestClient(app)
    response = client.post("/predict", json=_sample_product().model_dump(by_alias=True))

    assert response.status_code == 200
    body = response.json()
    assert body["score_percent"] >= 0
    assert "signals" in body
