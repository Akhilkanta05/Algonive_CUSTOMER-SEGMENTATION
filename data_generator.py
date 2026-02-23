import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

data = {
    "customer_id": np.random.randint(1000, 2000, n),
    "platform": np.random.choice(
        ["Google Ads", "Meta Ads", "LinkedIn Ads", "YouTube Ads", "TikTok Ads"],
        n
    ),
    "device": np.random.choice(
        ["Mobile", "Tablet", "Computer"], n
    ),
    "amount": np.random.randint(10, 500, n),
    "ad_spend": np.random.randint(5, 200, n),
    "watch_date": pd.date_range("2024-08-01", periods=n, freq="H")
}

df = pd.DataFrame(data)

df.to_csv("customers.csv", index=False)

print("Dataset created successfully!")