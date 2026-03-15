from sqlalchemy import text
from backend.database import engine

def test_connection():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            print("DB Version:", result.fetchone()[0])
            
            # Check PostGIS
            result = conn.execute(text("SELECT PostGIS_Version();"))
            print("PostGIS Version:", result.fetchone()[0])
            
            print("Successfully connected to the database!")
    except Exception as e:
        print("Database connection failed:", e)

if __name__ == "__main__":
    test_connection()
