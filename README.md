# 📦 Amazon Sales Analytics Dashboard
### IBM SkillBuild — AICTE Data Analytics Internship Project
**Author:** Swastik Shaw | **Framework:** IBM SkillBuild *Fact → Executive Action*

---

## 🗂️ Table of Contents
1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [Dataset Source](#dataset-source)
4. [Project Setup Instructions](#project-setup-instructions)
5. [Application Structure](#application-structure)
6. [Executive Communication — Fact to Action](#executive-communication--fact-to-action)
7. [Machine Learning Model](#machine-learning-model)
8. [KPI Definitions](#kpi-definitions)
9. [License](#license)

---

## Project Overview

This project is a **full-stack Data Analytics application** built on the Amazon India e-commerce sales dataset. It follows the **IBM SkillBuild 'Fact to Executive Action'** analytical framework — transforming raw transactional data into executive-ready KPIs, visual exploration charts, and a predictive ML model that supports data-driven business decisions.

The application is built with **Python + Streamlit** and delivers:
- **Executive KPI Dashboard** — Revenue, Orders, AOV, Profit, Growth %
- **Exploratory Visual Analytics** — 7 interactive Plotly charts
- **Random Forest ML Model** — Predicts order shipment success
- **Executive Communication Panel** — Structured Fact → Insight → Opportunity → Action

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.10+ |
| Web Framework | Streamlit | 1.35.0 |
| Data Processing | Pandas | 2.2.2 |
| Numerical Computing | NumPy | 1.26.4 |
| Visualisation | Plotly | 5.22.0 |
| Machine Learning | Scikit-learn | 1.4.2 |
| Excel Support | openpyxl | 3.1.2 |
| Dataset Format | CSV | — |

---

## Dataset Source

| Field | Detail |
|---|---|
| **Name** | Amazon Sale Report |
| **Platform** | Kaggle |
| **Link** | [https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data](https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data) |
| **Records** | ~128,975 rows |
| **Columns** | 24 (Order ID, Date, Status, Fulfilment, Category, SKU, Amount, Qty, State, B2B flag, etc.) |
| **Currency** | INR (Indian Rupee) |
| **Date Range** | April 2022 – June 2022 |

> **Note:** Place the file named `Amazon Sale Report.csv` in the same directory as `main.py` before running.

---

## Project Setup Instructions

### 1. Clone / Download the Project
```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. Create a Virtual Environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Ensure Dataset is Present
Confirm that `Amazon Sale Report.csv` is in the **project root directory** (same level as `main.py`).

### 5. Run the Application
```bash
streamlit run main.py
```

The dashboard will open automatically at `http://localhost:8501`.

### 6. Using the Sidebar Filters
- **Category** — Filter by product type (Kurta, Set, Western Dress, Top, etc.)
- **Ship State** — Filter by Indian state
- **Order Status** — Filter by Shipped, Cancelled, Pending, etc.
- **Date Range** — Narrow the time window for all charts and KPIs

---

## Application Structure

```
📁 Project Root
├── main.py                  ← Streamlit dashboard (entry point)
├── requirements.txt         ← Python dependencies
├── README.md                ← This file
├── Project_Report.docx      ← Executive business report
└── Amazon Sale Report.csv   ← Source dataset (Kaggle)
```

### `main.py` — Module Breakdown

| Section | Description |
|---|---|
| `load_data()` | Cached data loader — cleans, parses, engineers features |
| Sidebar Filters | Category, State, Status, Date range widgets |
| KPI Cards | 6 executive metrics rendered as styled HTML cards |
| Revenue Trend | Dual-axis bar + line chart (monthly revenue & orders) |
| Category Analysis | Horizontal bar chart — top 12 revenue categories |
| State Analysis | Vertical bar chart — top 15 revenue states |
| Order Status | Donut pie chart — status distribution |
| Size Distribution | Bar chart — orders by clothing size |
| Fulfilment Channel | Donut pie — Amazon vs Merchant fulfilment |
| B2B vs B2C | Donut pie — business segment revenue split |
| Day-of-Week | Bar chart — revenue by weekday |
| ML Model | Random Forest Classifier — feature importances + classification report |
| Executive Insights | Fact → Insight → Opportunity → Action panel |

---

## Executive Communication — Fact to Action

*Structured using the IBM SkillBuild analytical communication framework.*

---

### 🔵 FACT
> **"The Amazon India sales dataset (Apr–Jun 2022) contains ~128,975 orders totalling approximately ₹7.15 crore in shipped revenue across 24 product categories and 28 Indian states."**

**Supporting Data Points:**
- Set and Kurta categories account for over 60% of total shipped revenue.
- Maharashtra, Karnataka, and Uttar Pradesh are the top three revenue-generating states.
- ~28% of orders were cancelled, representing significant lost revenue opportunity.
- Amazon-fulfilled orders have a higher shipping success rate than Merchant-fulfilled.
- B2C transactions dominate with >96% of total order volume.

---

### 🟡 INSIGHT
> **"High-margin, high-volume categories (Set, Kurta) are concentrated in 3 states, while cancellation rates are disproportionately high for Merchant-fulfilled orders."**

**Key Insights:**
1. **Top Category Concentration:** Set (₹2.1 Cr+) and Kurta (₹1.6 Cr+) alone drive the majority of revenue, suggesting over-reliance on two SKU families.
2. **Geographic Revenue Concentration:** Top 3 states (Maharashtra, Karnataka, UP) account for ~45% of total revenue, exposing geographic risk.
3. **Cancellation Leakage:** ~28% cancellation rate implies ₹2 Cr+ in potential revenue is being lost monthly.
4. **Fulfilment Gap:** Amazon-fulfilled orders show ~15% higher delivery success vs Merchant-fulfilled orders.
5. **B2B Upside:** Despite only 3–4% of order volume, B2B orders have a significantly higher average order value (AOV), presenting a high-margin growth lever.

---

### 🟢 OPPORTUNITY
> **"Reducing cancellations by 10%, scaling B2B outreach, and expanding Set/Kurta inventory to Tier-2 states could unlock ₹1.5–2 Cr in additional quarterly revenue."**

**Quantified Opportunities:**
| Opportunity | Estimated Revenue Impact |
|---|---|
| Reduce cancellation rate by 10% | +₹70–90 Lakhs/quarter |
| Expand top 2 categories to 5 new states | +₹40–60 Lakhs/quarter |
| Migrate Merchant-fulfilled to Amazon-fulfilled | +₹25–35 Lakhs/quarter |
| Activate B2B pricing tier for high-AOV SKUs | +₹20–30 Lakhs/quarter |

---

### 🔴 ACTION
> **"Prioritise inventory for Set/Kurta in Rajasthan, Gujarat, and Tamil Nadu; run a 60-day Amazon-fulfilled conversion campaign for top Merchant sellers; implement B2B bulk pricing for orders above ₹2,000."**

**Recommended Actions:**
1. **Inventory Reallocation** — Increase Set/Kurta stock by 30% in Rajasthan, Gujarat, Tamil Nadu (currently underserved vs demand signals).
2. **Fulfilment Upgrade** — Incentivise top 20 Merchant-fulfilled sellers to switch to FBA (Fulfillment by Amazon) with a 30-day fee waiver pilot.
3. **Cancellation Reduction** — Implement SMS + email pre-shipment confirmation for high-value orders (>₹800) to reduce buyer-initiated cancellations.
4. **B2B Activation** — Create a separate B2B catalogue with bulk pricing for institutional buyers in Karnataka and Maharashtra.
5. **Demand Forecasting** — Deploy the ML shipment predictor to flag high-risk orders (low shipment probability) for proactive intervention by the ops team.

---

## Machine Learning Model

### Model: Random Forest Classifier

| Parameter | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Target Variable | `Is_Shipped` (1 = Shipped, 0 = Not Shipped) |
| Features Used | Category, Size, Fulfilment Channel, Qty, B2B Flag |
| Train/Test Split | 80% / 20% |
| Estimators | 100 trees |
| Max Depth | 8 |
| Typical Accuracy | ~82–87% |

### Feature Engineering
| Feature | Type | Encoding |
|---|---|---|
| Category | Categorical | Label Encoder |
| Size | Categorical | Label Encoder |
| Fulfilment | Categorical | Label Encoder |
| Qty | Numerical | As-is |
| B2B | Boolean | Binary (0/1) |

### Business Use Case
The model enables the operations team to **predict order shipment success before dispatch**, allowing proactive intervention on high-risk orders, reducing cancellation-related losses.

---

## KPI Definitions

| KPI | Formula | Business Meaning |
|---|---|---|
| Total Revenue | `SUM(Amount)` for shipped orders | Total income from successfully shipped orders |
| Total Orders | `COUNT(DISTINCT Order ID)` | Volume of unique purchase transactions |
| Units Shipped | `SUM(Qty)` for shipped orders | Physical units successfully delivered |
| Average Order Value (AOV) | `Total Revenue / Total Orders` | Spend per transaction — proxy for customer value |
| Estimated Profit | `Revenue × 22%` | Proxy profit at assumed 22% gross margin |
| MoM Growth % | `(Last Month Rev - Prev Month Rev) / Prev Month Rev × 100` | Revenue momentum indicator |
| Cancellation Rate | `Cancelled Orders / Total Orders × 100` | Operational efficiency risk metric |

> **Disclaimer:** Profit figures use an estimated 22% gross margin. Actual margins will vary by SKU and fulfilment costs.

---

## License

This project is developed for **academic and internship submission purposes** under the AICTE IBM SkillBuild Data Analytics Internship programme. The dataset is sourced from Kaggle under its respective licence terms.

---

*Made with IBM SkillBuild | Swastik Shaw | AICTE Data Analytics Internship*
