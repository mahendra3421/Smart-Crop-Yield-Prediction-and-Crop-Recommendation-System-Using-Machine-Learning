"""
generate_dataset.py
-------------------
Generates a realistic, research-grade synthetic dataset for Crop Yield Prediction.
Covers all required features: Crop, State, District, Season, Year, Area, Production,
Yield, Rainfall, Temperature, Humidity, pH, N, P, K, Pesticide_usage.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

CROPS = {
    "Rice":         dict(n=(80, 120), p=(40, 60),  k=(40, 60),  rain=(1000, 2500), temp=(22, 32), hum=(70, 90),  ph=(5.5, 7.0), base_yield=(3.0, 6.0)),
    "Wheat":        dict(n=(60, 100), p=(40, 60),  k=(40, 60),  rain=(400, 1000),  temp=(10, 25), hum=(40, 70),  ph=(6.0, 7.5), base_yield=(2.5, 5.0)),
    "Maize":        dict(n=(80, 120), p=(40, 70),  k=(40, 70),  rain=(600, 1200),  temp=(20, 30), hum=(50, 80),  ph=(5.8, 7.0), base_yield=(3.0, 6.5)),
    "Cotton":       dict(n=(60, 100), p=(25, 50),  k=(25, 50),  rain=(500, 1200),  temp=(21, 35), hum=(40, 75),  ph=(5.8, 8.0), base_yield=(1.5, 3.0)),
    "Sugarcane":    dict(n=(100,160), p=(50, 80),  k=(70, 110), rain=(1000, 2000), temp=(24, 35), hum=(65, 90),  ph=(6.0, 7.5), base_yield=(60, 110)),
    "Soybean":      dict(n=(20, 50),  p=(40, 60),  k=(40, 60),  rain=(600, 1200),  temp=(20, 30), hum=(50, 80),  ph=(6.0, 7.0), base_yield=(1.5, 3.5)),
    "Groundnut":    dict(n=(15, 40),  p=(30, 50),  k=(60, 90),  rain=(400, 900),   temp=(25, 35), hum=(50, 75),  ph=(5.5, 7.0), base_yield=(1.0, 3.0)),
    "Barley":       dict(n=(50, 90),  p=(35, 55),  k=(35, 55),  rain=(300, 800),   temp=(12, 25), hum=(35, 65),  ph=(6.0, 7.5), base_yield=(2.0, 4.5)),
    "Chickpea":     dict(n=(15, 40),  p=(40, 60),  k=(20, 40),  rain=(300, 700),   temp=(15, 30), hum=(40, 70),  ph=(6.0, 7.5), base_yield=(1.0, 2.5)),
    "Mustard":      dict(n=(50, 90),  p=(30, 50),  k=(30, 50),  rain=(250, 700),   temp=(10, 25), hum=(40, 70),  ph=(5.8, 7.5), base_yield=(1.5, 3.0)),
    "Potato":       dict(n=(80, 130), p=(60, 100), k=(100,150), rain=(500, 1000),  temp=(15, 25), hum=(65, 85),  ph=(5.0, 6.5), base_yield=(20, 40)),
    "Tomato":       dict(n=(80, 120), p=(60, 90),  k=(80, 120), rain=(600, 1200),  temp=(20, 30), hum=(60, 80),  ph=(6.0, 7.0), base_yield=(25, 60)),
    "Onion":        dict(n=(60, 100), p=(40, 70),  k=(60, 100), rain=(300, 700),   temp=(15, 28), hum=(50, 75),  ph=(6.0, 7.5), base_yield=(15, 35)),
    "Jowar":        dict(n=(50, 90),  p=(25, 50),  k=(25, 50),  rain=(400, 900),   temp=(25, 35), hum=(40, 70),  ph=(6.0, 7.5), base_yield=(1.0, 2.5)),
    "Bajra":        dict(n=(40, 80),  p=(20, 40),  k=(20, 40),  rain=(200, 600),   temp=(25, 38), hum=(30, 65),  ph=(5.5, 7.0), base_yield=(0.8, 2.2)),
    "Sunflower":    dict(n=(60, 100), p=(40, 60),  k=(40, 60),  rain=(400, 800),   temp=(20, 32), hum=(40, 70),  ph=(5.8, 7.5), base_yield=(1.0, 2.5)),
    "Turmeric":     dict(n=(60, 100), p=(40, 60),  k=(100,150), rain=(1200, 2500), temp=(20, 35), hum=(65, 90),  ph=(5.5, 7.0), base_yield=(6.0, 12.0)),
    "Ginger":       dict(n=(60, 100), p=(40, 60),  k=(80, 120), rain=(1500, 3000), temp=(20, 30), hum=(70, 90),  ph=(5.5, 6.5), base_yield=(10, 25)),
    "Tea":          dict(n=(80, 140), p=(20, 40),  k=(40, 80),  rain=(1500, 3000), temp=(15, 28), hum=(75, 95),  ph=(4.5, 5.5), base_yield=(1.5, 3.5)),
    "Coffee":       dict(n=(80, 140), p=(40, 70),  k=(80, 130), rain=(1500, 2500), temp=(18, 28), hum=(70, 90),  ph=(5.5, 6.5), base_yield=(1.0, 3.0)),
}

STATES_DISTRICTS = {
    "Uttar Pradesh":      ["Lucknow", "Agra", "Varanasi", "Kanpur", "Allahabad", "Bareilly"],
    "Maharashtra":        ["Pune", "Nagpur", "Nashik", "Aurangabad", "Kolhapur", "Amravati"],
    "Punjab":             ["Ludhiana", "Amritsar", "Jalandhar", "Patiala", "Bathinda"],
    "Madhya Pradesh":     ["Bhopal", "Indore", "Jabalpur", "Gwalior", "Ujjain", "Rewa"],
    "Gujarat":            ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Mehsana"],
    "Rajasthan":          ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Ajmer", "Bikaner"],
    "Andhra Pradesh":     ["Visakhapatnam", "Vijayawada", "Guntur", "Kurnool", "Kakinada"],
    "Tamil Nadu":         ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem"],
    "Karnataka":          ["Bengaluru", "Mysuru", "Hubli", "Mangaluru", "Belagavi"],
    "West Bengal":        ["Kolkata", "Howrah", "Durgapur", "Siliguri", "Asansol"],
    "Bihar":              ["Patna", "Gaya", "Muzaffarpur", "Bhagalpur", "Darbhanga"],
    "Haryana":            ["Gurugram", "Faridabad", "Ambala", "Hisar", "Rohtak"],
    "Odisha":             ["Bhubaneswar", "Cuttack", "Rourkela", "Sambalpur"],
    "Assam":              ["Guwahati", "Dibrugarh", "Silchar", "Jorhat"],
    "Telangana":          ["Hyderabad", "Warangal", "Khammam", "Nizamabad"],
}

SEASONS = ["Kharif", "Rabi", "Zaid", "Whole Year"]

SEASON_CROP_MAP = {
    "Kharif":     ["Rice", "Maize", "Cotton", "Sugarcane", "Soybean", "Groundnut", "Bajra", "Jowar", "Turmeric", "Ginger"],
    "Rabi":       ["Wheat", "Barley", "Chickpea", "Mustard", "Potato", "Onion", "Sunflower"],
    "Zaid":       ["Maize", "Sunflower", "Tomato", "Onion", "Groundnut"],
    "Whole Year": ["Sugarcane", "Tea", "Coffee", "Tomato", "Potato"],
}

N_ROWS = 8000


def _sample_crop_for_season(season: str) -> str:
    pool = SEASON_CROP_MAP.get(season, list(CROPS.keys()))
    return np.random.choice(pool)


def generate_dataset() -> pd.DataFrame:
    rows = []
    for _ in range(N_ROWS):
        state = np.random.choice(list(STATES_DISTRICTS.keys()))
        district = np.random.choice(STATES_DISTRICTS[state])
        season = np.random.choice(SEASONS)
        crop = _sample_crop_for_season(season)
        cfg = CROPS[crop]
        year = int(np.random.randint(2000, 2024))
        area = round(float(np.random.uniform(100, 50000)), 2)

        n   = round(float(np.random.uniform(*cfg["n"])),    2)
        p   = round(float(np.random.uniform(*cfg["p"])),    2)
        k   = round(float(np.random.uniform(*cfg["k"])),    2)
        rain = round(float(np.random.uniform(*cfg["rain"])), 2)
        temp = round(float(np.random.uniform(*cfg["temp"])), 2)
        hum  = round(float(np.random.uniform(*cfg["hum"])),  2)
        ph   = round(float(np.random.uniform(*cfg["ph"])),   2)
        pest = round(float(np.random.uniform(0.5, 20.0)),    2)

        # Yield is a function of N, P, K, rain, temp, pest — with noise
        n_norm   = (n   - cfg["n"][0])   / max(cfg["n"][1]   - cfg["n"][0],   1)
        p_norm   = (p   - cfg["p"][0])   / max(cfg["p"][1]   - cfg["p"][0],   1)
        k_norm   = (k   - cfg["k"][0])   / max(cfg["k"][1]   - cfg["k"][0],   1)
        r_norm   = (rain - cfg["rain"][0]) / max(cfg["rain"][1] - cfg["rain"][0], 1)
        t_norm   = 1.0 - abs((temp - (cfg["temp"][0] + cfg["temp"][1]) / 2) / max((cfg["temp"][1] - cfg["temp"][0]) / 2, 1))
        h_norm   = (hum - cfg["hum"][0]) / max(cfg["hum"][1] - cfg["hum"][0], 1)
        pest_pen = 1.0 - min(pest / 25.0, 0.25)   # slight penalty for over-pesticide

        base_lo, base_hi = cfg["base_yield"]
        base = base_lo + (base_hi - base_lo) * (
            0.25 * n_norm + 0.20 * p_norm + 0.20 * k_norm +
            0.20 * r_norm + 0.10 * t_norm + 0.05 * h_norm
        ) * pest_pen
        noise = np.random.normal(0, (base_hi - base_lo) * 0.08)
        yld = max(0.1, round(base + noise, 4))
        production = round(area * yld, 2)

        rows.append({
            "Crop":            crop,
            "State":           state,
            "District":        district,
            "Season":          season,
            "Year":            year,
            "Area":            area,
            "Production":      production,
            "Yield":           yld,
            "Rainfall":        rain,
            "Temperature":     temp,
            "Humidity":        hum,
            "pH":              ph,
            "N":               n,
            "P":               p,
            "K":               k,
            "Pesticide_usage": pest,
        })

    df = pd.DataFrame(rows)

    # Inject ~5% missing values in numeric columns to test imputation
    numeric_cols = ["Rainfall", "Temperature", "Humidity", "pH", "N", "P", "K",
                    "Pesticide_usage", "Area", "Production"]
    for col in numeric_cols:
        mask = np.random.rand(len(df)) < 0.05
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df = generate_dataset()
    path = os.path.join(out_dir, "crop_yield_dataset.csv")
    df.to_csv(path, index=False)
    print(f"Dataset saved → {path}  |  shape: {df.shape}")
    print(df.describe())
