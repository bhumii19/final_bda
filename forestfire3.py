from multiprocessing import Pool
import pandas as pd
import sqlite3

# Mapper function: returns (Month, Temperature)
def mapper(row):
    return (row["Month"], row["Temperature_Celsius"])

# Reducer: groups by Month and computes average temperature
def reducer(mapped_data):
    result = {}
    for month, temp in mapped_data:
        result.setdefault(month, []).append(temp)
    return {m: sum(v) / len(v) for m, v in result.items()}

# Runs MapReduce using multiprocessing
def run_mapreduce(df):
    with Pool() as p:
        mapped = p.map(mapper, [row for _, row in df.iterrows()])
    reduced = reducer(mapped)
    print("\n🌡️ Average Temperature per Month:")
    for m, t in reduced.items():
        print(f"{m}: {t:.2f}")
    return reduced

# Finds top months with largest fire areas
def top_fire_months(df, top_n=5):
    top = (
        df.groupby("Month")["Burned_Area_hectares"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )
    print(f"\n🔥 Top {top_n} Months with Largest Fire Area:")
    print(top)
    return top

# Computes correlation between temperature and burned area
def temperature_area_correlation(df):
    corr = df["Temperature_Celsius"].corr(df["Burned_Area_hectares"])
    print(f"\n📊 Correlation between Temperature and Fire Area: {corr:.2f}")
    return corr

# SQL query to calculate average burned area by month
def query_avg_area_by_month(conn):
    query = """
        SELECT Month, AVG(Burned_Area_hectares) AS avg_area
        FROM forestfires
        GROUP BY Month
        ORDER BY avg_area DESC;
    """
    result = pd.read_sql_query(query, conn)
    print("\n🧾 Average Burned Area by Month (from SQL):")
    print(result)
    return result

# Main function to run the entire pipeline
def run_pipeline():
    print("=== 🌲 Forest Fire Analysis Pipeline Started ===\n")

    # Step 1: Read dataset
    df = pd.read_csv("forestfires.csv")
    print(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns.\n")

    # Step 2: Store in SQLite database
    conn = sqlite3.connect("forestfires.db")
    df.to_sql("forestfires", conn, if_exists="replace", index=False)
    print("✅ Data saved to SQLite database.\n")

    # Step 3: MapReduce
    run_mapreduce(df)

    # Step 4: Top months with largest burned area
    top_fire_months(df)

    # Step 5: Correlation between temperature and burned area
    temperature_area_correlation(df)

    # Step 6: SQL query result
    query_avg_area_by_month(conn)

    print("\n=== ✅ Pipeline Completed Successfully ===")

# Entry point
if __name__ == "__main__":
    run_pipeline()
