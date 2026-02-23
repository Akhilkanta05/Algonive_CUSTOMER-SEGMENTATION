# 👥 Customer Segmentation Dashboard

An interactive **Streamlit** dashboard for customer segmentation and e-commerce analytics using **RFM Analysis**, **K-Means Clustering**, and rich **Plotly** visualisations.

---

## 📌 Features

- **RFM Analysis** – Recency, Frequency, and Monetary scoring for every customer
- **K-Means Clustering** – Automatic customer segmentation into meaningful groups
- **Interactive Visualisations** – Bar charts, scatter plots, heatmaps, and more powered by Plotly
- **Streamlit UI** – Clean, responsive single-page dashboard

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Akhilkanta05/Algonive_CUSTOMER-SEGMENTATION.git
cd Algonive_CUSTOMER-SEGMENTATION

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📂 Project Structure

```
├── app.py              # Main Streamlit application
├── data_generator.py   # Synthetic data generation helper
├── customers.csv       # Sample customer dataset
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── config.toml     # Streamlit theme configuration
└── README.md
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Streamlit** | Web application framework |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **Plotly** | Interactive charts |
| **Scikit-learn** | K-Means clustering |
| **Matplotlib** | Gradient heatmap styling |
| **OpenPyXL** | Excel file support |

---

## 📊 Dataset

Upload any transactional e-commerce CSV/Excel with customer IDs, invoice dates, quantities, and unit prices to derive RFM metrics and automatic customer segments.

---

## 📄 License

This project is developed as part of the **Algonive** internship program.

---

## 👤 Author

**Akhil Kanta**  
[GitHub](https://github.com/Akhilkanta05)
