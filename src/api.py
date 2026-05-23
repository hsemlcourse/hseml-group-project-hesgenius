"""FastAPI application for product quality scoring."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from src.inference import PredictionResponse, ProductInput, dump_product, estimate_quality


app = FastAPI(
    title="Marketplace Quality API",
    version="1.0.0",
    description="API for scoring marketplace product cards with a transparent baseline.",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "Marketplace Quality API is running.",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(product: ProductInput) -> PredictionResponse:
    try:
        return estimate_quality(product)
    except Exception as exc:  # pragma: no cover - defensive API boundary
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/batch_predict")
def batch_predict(products: list[ProductInput]) -> dict[str, list[dict[str, Any]]]:
    results = []
    for product in products:
        prediction = estimate_quality(product)
        results.append(
            {
                "input": dump_product(product),
                "prediction": prediction.model_dump() if hasattr(prediction, "model_dump") else prediction.dict(),
            }
        )
    return {"results": results}


@app.get("/example")
def example_payload() -> dict[str, Any]:
    example = ProductInput(
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
    prediction = estimate_quality(example)
    return {
        "input": dump_product(example),
        "prediction": prediction.model_dump() if hasattr(prediction, "model_dump") else prediction.dict(),
    }
