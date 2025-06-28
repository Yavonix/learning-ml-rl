import polars as pl
import polars.selectors as cs

cat_columns = {
    "MSSubClass": pl.Categorical,
    "MSZoning": pl.Categorical,
    "Street": pl.Categorical,
    "Alley": pl.Categorical,
    "LotShape": pl.Categorical,
    "LandContour": pl.Categorical,
    "Utilities": pl.Categorical,
    "LotConfig": pl.Categorical,
    "LandSlope": pl.Categorical,
    "Neighborhood": pl.Categorical,
    "Condition1": pl.Categorical,
    "Condition2": pl.Categorical,
    "BldgType": pl.Categorical,
    "HouseStyle": pl.Categorical,
    "OverallQual": pl.Categorical,
    "OverallCond": pl.Categorical,
    "RoofStyle": pl.Categorical,
    "RoofMatl": pl.Categorical,
    "Exterior1st": pl.Categorical,
    "Exterior2nd": pl.Categorical,
    "MasVnrType": pl.Categorical,
    "ExterQual": pl.Categorical,
    "ExterCond": pl.Categorical,
    "Foundation": pl.Categorical,
    "BsmtQual": pl.Categorical,
    "BsmtCond": pl.Categorical,
    "BsmtExposure": pl.Categorical,
    "BsmtFinType1": pl.Categorical,
    "BsmtFinType2": pl.Categorical,
    "Heating": pl.Categorical,
    "HeatingQC": pl.Categorical,
    "CentralAir": pl.Categorical,
    "Electrical": pl.Categorical,
    "KitchenQual": pl.Categorical,
    "Functional": pl.Categorical,
    "FireplaceQu": pl.Categorical,
    "GarageType": pl.Categorical,
    "GarageFinish": pl.Categorical,
    "GarageQual": pl.Categorical,
    "GarageCond": pl.Categorical,
    "PavedDrive": pl.Categorical,
    "PoolQC": pl.Categorical,
    "Fence": pl.Categorical,
    "MiscFeature": pl.Categorical,
    "SaleType": pl.Categorical,
    "SaleCondition": pl.Categorical,
}

def process_df(df: pl.DataFrame):
    df = df.drop("Id")

    df = df.with_columns(
        (pl.col("GarageYrBlt").is_null().alias("HasGarage")),
        (pl.col("YrSold") - pl.col("GarageYrBlt"))
        .alias("GarageYrBlt"),
        pl.col("SalePrice").log()
    )

    df = df.with_columns(
        ((cs.numeric() - cs.numeric().mean()) / cs.numeric().std()),
    )

    df = df.fill_null(0) # We fill LotFrontage and MasVnrArea NA with 0

    df = df.to_dummies(columns=cs.categorical())

    df = df.select(
        pl.all().exclude("SalePrice"),
        pl.col("SalePrice")
    )

    return df

def get_test_df():
    raw_test_df = pl.read_csv("./train.csv",
                     null_values="NA",
                     schema_overrides=cat_columns)
    
    test_df = process_df(raw_test_df)

    return test_df

get_test_df()