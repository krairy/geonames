from transliterate import slugify, detect_language
import pandas as pd


def connect_to_db():
    from dotenv import load_dotenv
    import os

    assert load_dotenv()

    from sqlalchemy import create_engine
    from sqlalchemy.engine.url import URL
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.sql import text

    USR = os.getenv("USR")  # FOR GODS' SAKE NEVER CALL an env variable "USERNAME"
    PWD = os.getenv("PWD")
    DB_HOST = os.getenv("DB_HOST")
    PORT = os.getenv("PORT")
    DB = os.getenv("DB")

    DATABASE = {
        "drivername": "postgresql",
        "username": USR,
        "password": PWD,
        "host": DB_HOST,
        "port": PORT,
        "database": DB,
        "query": {},
    }

    # Creating an Engine object
    engine = create_engine(URL.create(**DATABASE))

    # Checking the connetion
    try:
        with engine.connect() as conn:
            # Trying to execute a simple test query. The `text` function converst a string into and SQL-query
            result = conn.execute(text("SELECT 1"))
            for _ in result:
                pass  # don't do anything
        print(f"Connection established: {DATABASE['database']} на {DATABASE['host']}")
    except SQLAlchemyError as e:
        print(f"Connection error: {e}")


def main(query, low_memory: bool = True, weight_mode: str = "exp"):
    """
    Function that searches city names similar to the query string. Takes alternative names into account.

    Parameters
    ----------
    query : the query string

    weight_mode={None, 'sq', 'exp'}, default 'exp'
        * None: do not weight closer matches
        * 'sq': apply parabolic weighting to the similarity scores
        * 'exp': apply exponential weighting to the similarity scores

    """


def search_exp_lowmem(query, k=10):
    """
    Perform a low-memory search for cities based on a given query.

    Parameters:
    - query (str): The search query, representing the city name or an alternative name.
    - k (int, optional): The number of top matching cities to retrieve. Defaults to 10.

    Returns:
    - pd.DataFrame: A DataFrame containing information about the top matching cities,
      including geonameid, country_code, name, asciiname, population, and Country.
      The DataFrame is sorted by the matching score in descending order.

    The function uses a weighted similarity scoring mechanism based on the exponential
    sum of similarity scores for alternative city names. The scores are calculated using
    the Levenshtein distance and then transformed with the exponential function.
    The top k cities with the highest scores are retrieved from the 'cities15000' dataset
    and joined with additional information from the 'countryInfo' dataset.

    Example:
    >>> result_df = search_exp_lowmem("New York", k=5)
    >>> print(result_df)

    Note:
    This function assumes the existence of a 'cities15000' table with columns
    geonameid, country_code, name, and asciiname, and a 'countryInfo' table with
    columns ISO and Country. The database connection 'engine' should be available.
    """
    # Function implementation...
    if detect_language(query) is not None:
        query = slugify(query)
    scores = {}  # container for match scores for each city

    for (
        ind,
        name_list,
    ) in (
        d.items()
    ):  ## for each city calculate similarity scores with evry alternative name
        _ = np.array(
            process.extract(query, name_list)
        )  # so that the exponent is not too large!
        # Calculate the function
        scores[ind] = np.exp(_[:, 1].astype(int)).sum() / len(_)
        scores[ind] = np.log(scores[ind])  # sum up the exponents of the scores...

    # sorted by the matching score (.2 ms faster with the native Python function)
    scores_df = pd.DataFrame.from_records(
        sorted(scores.items(), key=lambda item: item[1], reverse=True),
        columns=["geonameid", "score"],
    )

    indexes = tuple(
        scores_df.loc[:k, "geonameid"]
    )  # select the DataFrame indicies of the top k

    query = f"""
        SELECT geonameid, country_code, name, asciiname, population, ci."Country"
        FROM cities15000  
        LEFT JOIN (SELECT "ISO", "Country" FROM "countryInfo") AS ci
        ON cities15000.country_code = ci."ISO"
        WHERE geonameid IN {indexes};
    """
    qres = pd.read_sql_query(query, con=engine).drop_duplicates()

    return pd.merge(qres, scores_df, on="geonameid", how="right")
