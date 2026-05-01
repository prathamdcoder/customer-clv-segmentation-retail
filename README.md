# Customer Segmentation & Lifetime Value Prediction for Retail

## What this project does
Builds a full customer analytics pipeline on 797,000+ real retail transactions to:
1. **Segment** customers into behavioural groups (Champions, Loyal, At-Risk, Dormant)
2. **Predict** 12-month revenue per customer (Customer Lifetime Value)
3. **Identify** customers at risk of churning before they lapse

## Dataset
**UCI Online Retail II** — Real transactional data from a UK-based online retailer
- Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii
- Period: Dec 2009 – Dec 2011
- Size: ~797,000 transactions, ~4,000 customers

The code auto-downloads via `ucimlrepo`. If unavailable, generates a statistically equivalent synthetic dataset.

## Business Questions Answered
> "Which customers should we prioritise for retention spend?"
> "How much revenue will each customer generate in the next 12 months?"
> "Which high-value customers are about to churn?"

## Key Results
- Top 15% of customers → 50%+ of projected 12-month revenue
- Churn prediction AUC: ~0.98 (near-perfect ranking of at-risk customers)
- 5 distinct behavioural segments identified
- Loyal/engaged customers spend ~1,000%+ more than dormant ones

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib lifetimes openpyxl ucimlrepo
python clv_analysis.py
```

## What the code does (step by step)

| Step | What happens | Tool |
|------|-------------|------|
| 1 | Load UCI Online Retail II dataset | ucimlrepo / pandas |
| 2 | Clean data: remove cancellations, nulls, invalid rows | pandas |
| 3 | Aggregate to customer-level via SQL query | sqlite3 |
| 4 | Compute RFM scores (Recency, Frequency, Monetary) | pandas |
| 5 | Find optimal number of clusters (silhouette score) | scikit-learn |
| 6 | Segment customers using K-Means | scikit-learn |
| 7 | Predict 12-month purchase frequency (BG/NBD model) | lifetimes |
| 8 | Predict average transaction spend (Gamma-Gamma model) | lifetimes |
| 9 | Compute CLV = frequency × spend | pandas |
| 10 | Train churn prediction model (Logistic Regression) | scikit-learn |
| 11 | Build 4-panel analytics dashboard | matplotlib |

## Key Concepts Explained Simply

**RFM**: Three dimensions that define customer value — how recently they bought, how often, and how much. Simple but powerful ranking framework.

**K-Means**: Groups customers by similarity in RFM scores. The algorithm finds natural clusters — customers within a cluster are treated with the same marketing strategy.

**BG/NBD (Purchase Model)**: "I used a probabilistic model that looks at how often a customer bought historically and when they last bought — and estimates how many times they'll buy in the next year."

**Gamma-Gamma (Spend Model)**: "A companion model that predicts the average spend per purchase for each customer, based on their historical spending pattern."

**Churn Model (Logistic Regression)**: Outputs a probability score for each customer — high score = likely to churn. Enables proactive outreach to retain high-CLV customers.

## Output
- `clv_dashboard.png`: 4-panel analytics dashboard

## Skills Demonstrated
`Python` `pandas` `SQL (sqlite3)` `scikit-learn` `lifetimes` `K-Means` `Logistic Regression` `BG/NBD` `Gamma-Gamma` `matplotlib` `Customer Analytics` `Feature Engineering`
