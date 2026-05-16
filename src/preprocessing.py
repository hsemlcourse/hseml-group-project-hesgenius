"""
Предобработка данных перед созданием baseline-модели, здесь будем
чистить датасет, нормализовать данные, добавлять новые признаки 
и таргет из EDA, убирать метрики с угрозой даталика, а затем
сохраним все в новый предобработанный датасет в формате .csv
в папку /data/processed.
"""

from typing import Iterable

import numpy as np
import pandas as pd


RANDOM_SEED = 42

LEAKAGE_COLUMNS = [
    "Рейтинг",
    "Оценок",
    "rating_norm",
    "reviews_log1p",
    "reviews_log_norm",
    "success_proxy_score",
]

ID_URL_COLUMNS = [
    "Артикул поставщика",
    "URL",
    "Картинка",
    "Галерея",
    "Видеообложка",
    "Страница продавца",
]

MASKED_COLUMNS = [
    "ID продавца",
    "Рейтинг продавца",
    "Отзывы продавца",
    "Зарегистрирован",
    "Возраст магазина",
    "Продажи продавца",
    "Процент выкупа",
    "Юр. лицо",
    "ИНН",
    "ОГРН / ОГРНИП",
    "Адрес продавца",
    "Оплата за отзыв",
]

BASE_FEATURE_COLUMNS = [
    "Артикул",
    "Бренд",
    "Название",
    "Категория",
    "Спец. цена",
    "Цена",
    "Старая цена",
    "Скидка",
    "Цвет",
    "Размер",
    "Реклама",
    "Продавец",
    "Фулфилмент",
    "Доставка",
    "Сезон",
    "Название размера",
    "Кол-во картинок",
    "Описание",
    "Rich-контент",
    "Состав",
    "Страна производства",
]

ENGINEERED_FEATURE_COLUMNS = [
    "name_len",
    "name_word_count",
    "description_len",
    "description_word_count",
    "description_missing",
    "has_brand",
    "has_country",
    "has_video",
    "has_rich_content",
    "is_ad",
    "price_log1p",
    "special_price_log1p",
    "old_price_log1p",
    "discount_ratio_calc",
    "price_to_old_price_ratio",
    "images_count_log1p",
    "has_images",
    "many_images",
    "brand_frequency",
    "seller_frequency",
    "category_frequency",
]

NUMERIC_FEATURE_COLUMNS = [
    "Спец. цена",
    "Цена",
    "Старая цена",
    "Скидка",
    "Кол-во картинок",
    "name_len",
    "name_word_count",
    "description_len",
    "description_word_count",
    "description_missing",
    "has_brand",
    "has_country",
    "has_video",
    "has_rich_content",
    "is_ad",
    "price_log1p",
    "special_price_log1p",
    "old_price_log1p",
    "discount_ratio_calc",
    "price_to_old_price_ratio",
    "images_count_log1p",
    "has_images",
    "many_images",
    "brand_frequency",
    "seller_frequency",
    "category_frequency",
]

TEXT_FEATURE_COLUMNS = ["Название", "Описание", "Состав"]

CATEGORICAL_FEATURE_COLUMNS = [
    "Бренд",
    "Категория",
    "Цвет",
    "Размер",
    "Реклама",
    "Продавец",
    "Фулфилмент",
    "Доставка",
    "Сезон",
    "Название размера",
    "Rich-контент",
    "Страна производства",
]


OUTPUT_DIR = "./data/processed/products_processed.csv"
INPUT_DIR = "./data/raw/products.csv"


def normalize_empty_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Нормализуем пустые ячейки
    """
    result = df.copy()
    empty_like = {
        "",
        " ",
        "-",
        "--",
        "nan",
        "NaN",
        "none",
        "None",
        "null",
        "Null",
        "N/A",
        "n/a",
    }

    for col in result.columns:
        if pd.api.types.is_object_dtype(result[col]):
            result[col] = result[col].astype("string").str.strip()
            result[col] = result[col].replace(list(empty_like), pd.NA)

    return result


def yes_no_to_int(series: pd.Series) -> pd.Series:
    """
    Переводим категории Да/Нет в бинарный формат 0/1
    """
    return series.map({"Да": 1, "Нет": 0}).astype("float")


def safe_log1p(series: pd.Series) -> pd.Series:
    return np.log1p(series.clip(lower=0))


def remove_invalid_cards(
    df: pd.DataFrame,
    min_price: float = 1.0,
    min_name_len: int = 3,
    max_price_quantile: float = 0.995,
) -> tuple[pd.DataFrame, dict]:
    """
    Удаляем ненужные карточки
    """
    result = df.copy()
    initial_rows = len(result)

    invalid_reason = pd.Series("", index=result.index, dtype="object")

    duplicate_mask = result.duplicated()
    invalid_reason.loc[duplicate_mask] += "|duplicate_row"

    missing_article = result["Артикул"].isna()
    invalid_reason.loc[missing_article] += "|missing_article"

    missing_name = result["Название"].isna() | (
        result["Название"].fillna("").astype(str).str.len() < min_name_len
    )
    invalid_reason.loc[missing_name] += "|invalid_name"

    invalid_price = result["Цена"].isna() | (result["Цена"] < min_price)
    invalid_reason.loc[invalid_price] += "|invalid_price"

    price_cap = result["Цена"].quantile(max_price_quantile)
    extreme_price = result["Цена"] > price_cap
    invalid_reason.loc[extreme_price] += "|extreme_price"

    invalid_images = result["Кол-во картинок"].isna() | (result["Кол-во картинок"] < 0)
    invalid_reason.loc[invalid_images] += "|invalid_images_count"

    invalid_rating = result["Рейтинг"].notna() & ~result["Рейтинг"].between(1, 5)
    invalid_reason.loc[invalid_rating] += "|invalid_rating_range"

    invalid_reviews = result["Оценок"].notna() & (result["Оценок"] < 0)
    invalid_reason.loc[invalid_reviews] += "|invalid_reviews_count"

    invalid_mask = invalid_reason.ne("")

    clean = result.loc[~invalid_mask].copy()

    return clean


def add_regression_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Формируем таргет-метрику в новой колонке
    """
    result = df.copy()

    result["rating_norm"] = (result["Рейтинг"] - 1) / 4

    max_reviews = result["Оценок"].max(skipna=True)
    if pd.isna(max_reviews) or max_reviews <= 0:
        result["reviews_log1p"] = np.nan
        result["reviews_log_norm"] = np.nan
    else:
        result["reviews_log1p"] = np.log1p(result["Оценок"])
        result["reviews_log_norm"] = result["reviews_log1p"] / np.log1p(max_reviews)

    result["success_proxy_score"] = (
        0.65 * result["rating_norm"]
        + 0.35 * result["reviews_log_norm"]
    )

    return result


def add_eda_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Здесь добавляем новые признаки, полученные при анализе данных
    """
    result = df.copy()

    result["name_len"] = result["Название"].fillna("").astype(str).str.len()
    result["name_word_count"] = (
        result["Название"].fillna("").astype(str).str.split().str.len()
    )

    result["description_len"] = result["Описание"].fillna("").astype(str).str.len()
    result["description_word_count"] = (
        result["Описание"].fillna("").astype(str).str.split().str.len()
    )
    result["description_missing"] = result["Описание"].isna().astype(int)

    result["has_brand"] = result["Бренд"].notna().astype(int)
    result["has_country"] = result["Страна производства"].notna().astype(int)
    result["has_video"] = result["Видеообложка"].notna().astype(int)
    result["has_rich_content"] = yes_no_to_int(result["Rich-контент"]).fillna(0).astype(int)
    result["is_ad"] = yes_no_to_int(result["Реклама"]).fillna(0).astype(int)

    result["price_log1p"] = safe_log1p(result["Цена"])
    result["special_price_log1p"] = safe_log1p(result["Спец. цена"])
    result["old_price_log1p"] = safe_log1p(result["Старая цена"])

    result["discount_ratio_calc"] = np.where(
        result["Старая цена"].fillna(0) > 0,
        (result["Старая цена"] - result["Цена"]) / result["Старая цена"],
        np.nan,
    )
    result["discount_ratio_calc"] = result["discount_ratio_calc"].clip(lower=-1, upper=1)

    result["price_to_old_price_ratio"] = np.where(
        result["Старая цена"].fillna(0) > 0,
        result["Цена"] / result["Старая цена"],
        np.nan,
    )
    result["price_to_old_price_ratio"] = result["price_to_old_price_ratio"].clip(
        lower=0, upper=3
    )

    result["images_count_log1p"] = safe_log1p(result["Кол-во картинок"])
    result["has_images"] = (result["Кол-во картинок"].fillna(0) > 0).astype(int)
    result["many_images"] = (result["Кол-во картинок"].fillna(0) >= 10).astype(int)

    for source_col, feature_col in [
        ("Бренд", "brand_frequency"),
        ("Продавец", "seller_frequency"),
        ("Категория", "category_frequency"),
    ]:
        frequencies = result[source_col].value_counts(normalize=True, dropna=False)
        result[feature_col] = result[source_col].map(frequencies).astype(float)

    return result


def normalize_feature_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Нормализуем пропущенные значения и категориальные признаки
    """
    result = df.copy()

    for col in CATEGORICAL_FEATURE_COLUMNS:
        if col not in result.columns:
            continue
        result[col] = (
            result[col]
            .astype("string")
            .fillna("__MISSING__")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    for col in TEXT_FEATURE_COLUMNS:
        if col not in result.columns:
            continue
        result[col] = result[col].astype("string").fillna("")
        result[col] = result[col].str.strip().str.replace(r"\s+", " ", regex=True)

    for col in NUMERIC_FEATURE_COLUMNS:
        if col not in result.columns:
            continue
        median = result[col].median(skipna=True)
        fill_value = 0.0 if pd.isna(median) else float(median)
        result[col] = result[col].fillna(fill_value)

    return result


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращаем обработанные данные без колонок, которые могут вызвать даталик
    """
    ordered_columns = (
        ["Артикул", "success_proxy_score"]
        + [col for col in BASE_FEATURE_COLUMNS if col != "Артикул"]
        + ENGINEERED_FEATURE_COLUMNS
    )

    existing_columns = []
    for col in ordered_columns:
        if col in df.columns and col not in existing_columns:
            existing_columns.append(col)

    output = df[existing_columns].copy()

    forbidden = set(LEAKAGE_COLUMNS) - {"success_proxy_score"}
    leaked = [col for col in output.columns if col in forbidden]
    if leaked:
        raise ValueError(f"Поля с даталиком: {leaked}")

    return output


def build_processed_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:

    cleaned = remove_invalid_cards(raw_df)
    with_target = add_regression_target(cleaned)
    with_features = add_eda_features(with_target)
    normalized = normalize_feature_values(with_features)
    features_all = select_output_columns(normalized)
    processed = features_all[features_all["success_proxy_score"].notna()].copy()

    return processed


def save_outputs(
    processed: pd.DataFrame
) -> None:

    processed.to_csv(OUTPUT_DIR, index=False)

    available_numeric = [col for col in NUMERIC_FEATURE_COLUMNS if col in processed.columns]
    available_categorical = [col for col in CATEGORICAL_FEATURE_COLUMNS if col in processed.columns]
    available_text = [col for col in TEXT_FEATURE_COLUMNS if col in processed.columns]


    print(f"Сохранено итоговых строк: {OUTPUT_DIR} ({processed.shape})")


def main() -> None:
    raw_df = pd.read_csv(INPUT_DIR)
    print(f"Размер исходного датасета: {raw_df.shape[0]} строк, {raw_df.shape[1]} колонок")
    processed = build_processed_dataset(raw_df)
    save_outputs(processed)


if __name__ == "__main__":
    main()
