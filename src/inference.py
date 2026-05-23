"""Shared scoring logic for API and Streamlit UI.

This project does not currently ship a serialized trained model, so the app
exposes a transparent baseline scorer that can be queried through FastAPI and
used directly in Streamlit.
"""

from __future__ import annotations

from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover - compatibility fallback
    from pydantic import BaseModel, Field

    ConfigDict = None  # type: ignore[assignment]
    PYDANTIC_V2 = False


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "nan"}:
        return ""
    return text


def _bool_score(value: Any) -> float:
    text = _clean_text(value).lower()
    if not text:
        return 0.0
    if text in {"да", "yes", "y", "true", "1", "есть"}:
        return 1.0
    if text in {"нет", "no", "n", "false", "0", "нету"}:
        return 0.0
    return 1.0


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _scale(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    return _clip((value - lower) / (upper - lower))


def _score_title(title: str) -> float:
    words = [word for word in title.split() if word]
    length_score = _scale(len(title), 15.0, 90.0)
    word_score = _scale(len(words), 3.0, 10.0)
    return 0.55 * length_score + 0.45 * word_score


def _score_description(description: str) -> float:
    words = [word for word in description.split() if word]
    length_score = _scale(len(description), 60.0, 1200.0)
    word_score = _scale(len(words), 8.0, 180.0)
    return 0.6 * length_score + 0.4 * word_score


def _score_price(price: float, old_price: float | None, special_price: float | None) -> float:
    if old_price and old_price > 0 and price >= 0:
        discount = _clip((old_price - price) / old_price, 0.0, 1.0)
        return 0.35 + 0.65 * discount

    if special_price and special_price > 0 and price > 0:
        ratio = _clip(special_price / price, 0.0, 3.0)
        return _clip(1.15 - 0.4 * ratio, 0.0, 1.0)

    if price <= 0:
        return 0.0

    if price < 1500:
        return 0.8
    if price < 5000:
        return 0.95
    if price < 12000:
        return 0.75
    return 0.5


def _score_reviews(rating: float | None, reviews: float | None) -> float:
    if rating is None and reviews is None:
        return 0.5

    rating_score = 0.5 if rating is None else _clip((rating - 1.0) / 4.0)
    if reviews is None or reviews <= 0:
        review_score = 0.35
    else:
        review_score = _clip((reviews ** 0.5) / 35.0)
    return 0.7 * rating_score + 0.3 * review_score


class ProductInput(BaseModel):
    article: str = Field(..., alias="Артикул")
    name: str = Field(..., alias="Название")
    description: str = Field("", alias="Описание")
    brand: str | None = Field(None, alias="Бренд")
    category: str | None = Field(None, alias="Категория")
    price: float = Field(..., ge=0, alias="Цена")
    special_price: float | None = Field(None, ge=0, alias="Спец. цена")
    old_price: float | None = Field(None, ge=0, alias="Старая цена")
    discount: float | None = Field(None, ge=0, alias="Скидка")
    color: str | None = Field(None, alias="Цвет")
    size: str | None = Field(None, alias="Размер")
    seller: str | None = Field(None, alias="Продавец")
    fulfillment: str | None = Field(None, alias="Фулфилмент")
    delivery: str | None = Field(None, alias="Доставка")
    season: str | None = Field(None, alias="Сезон")
    size_name: str | None = Field(None, alias="Название размера")
    images_count: int | None = Field(0, ge=0, alias="Кол-во картинок")
    rich_content: str | None = Field(None, alias="Rich-контент")
    country: str | None = Field(None, alias="Страна производства")
    video_cover: str | None = Field(None, alias="Видеообложка")
    advertising: str | None = Field(None, alias="Реклама")
    rating: float | None = Field(None, ge=1, le=5, alias="Рейтинг")
    reviews: float | None = Field(None, ge=0, alias="Оценок")

    if PYDANTIC_V2:
        model_config = ConfigDict(populate_by_name=True, extra="ignore")
    else:  # pragma: no cover - compatibility fallback
        class Config:
            allow_population_by_field_name = True
            extra = "ignore"


class PredictionResponse(BaseModel):
    score: float
    score_percent: int
    grade: str
    explanation: str
    signals: dict[str, float]
    recommendations: list[str]


def _recommendations(product: ProductInput, signals: dict[str, float]) -> list[str]:
    tips: list[str] = []
    if signals["title_quality"] < 0.5:
        tips.append("Сделайте название длиннее и конкретнее: добавьте модель, тип и ключевые характеристики.")
    if signals["description_quality"] < 0.45:
        tips.append("Добавьте подробное описание с материалами, размерами и преимуществами.")
    if signals["image_quality"] < 0.5:
        tips.append("Увеличьте количество фотографий и добавьте фото товара крупным планом.")
    if signals["catalog_quality"] < 0.5:
        tips.append("Заполните бренд, страну производства и Rich-контент, если они доступны.")
    if signals["promotion_quality"] < 0.4:
        tips.append("Проверьте рекламную маркировку и ценовую политику карточки.")
    if product.rating is not None and signals["review_quality"] < 0.5:
        tips.append("Поддерживайте рейтинг и количество отзывов через качество сервиса и упаковки.")

    if not tips:
        tips.append("Карточка выглядит сбалансированной по базовым сигналам качества.")

    return tips


def estimate_quality(product: ProductInput) -> PredictionResponse:
    title_quality = _score_title(_clean_text(product.name))
    description_quality = _score_description(_clean_text(product.description))
    image_quality = _scale(float(product.images_count or 0), 0.0, 8.0)
    catalog_quality = (
        0.25 * _bool_score(product.brand)
        + 0.2 * _bool_score(product.category)
        + 0.2 * _bool_score(product.country)
        + 0.2 * _bool_score(product.rich_content)
        + 0.15 * _bool_score(product.video_cover)
    )
    price_quality = _score_price(float(product.price), product.old_price, product.special_price)
    review_quality = _score_reviews(product.rating, product.reviews)
    promotion_quality = 1.0 - _bool_score(product.advertising) * 0.35
    promotion_quality = _clip(promotion_quality)

    score = (
        0.2 * title_quality
        + 0.18 * description_quality
        + 0.16 * image_quality
        + 0.16 * catalog_quality
        + 0.15 * price_quality
        + 0.15 * review_quality
        + 0.05 * promotion_quality
    )
    score = _clip(score)

    if score >= 0.75:
        grade = "Высокое качество"
    elif score >= 0.5:
        grade = "Среднее качество"
    else:
        grade = "Нужна доработка"

    signals = {
        "title_quality": round(title_quality, 3),
        "description_quality": round(description_quality, 3),
        "image_quality": round(image_quality, 3),
        "catalog_quality": round(catalog_quality, 3),
        "price_quality": round(price_quality, 3),
        "review_quality": round(review_quality, 3),
        "promotion_quality": round(promotion_quality, 3),
    }
    recommendations = _recommendations(product, signals)
    explanation = (
        f"Итоговая оценка {grade.lower()} складывается из качества текста, изображений, "
        f"ценовой логики и заполненности карточки."
    )

    return PredictionResponse(
        score=round(score, 4),
        score_percent=int(round(score * 100)),
        grade=grade,
        explanation=explanation,
        signals=signals,
        recommendations=recommendations,
    )


def dump_product(product: ProductInput) -> dict[str, Any]:
    if hasattr(product, "model_dump"):
        return product.model_dump(by_alias=True)
    return product.dict(by_alias=True)
