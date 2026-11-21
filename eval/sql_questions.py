"""Hand-written SQL questions with reference SQL. Gold answers are computed by running the
reference SQL against the warehouse, so they always match the data."""

SQL_QUESTIONS = [
    ("What was Walmart's total revenue in fiscal 2026?",
     "SELECT revenue FROM financials WHERE ticker='WMT' AND fiscal_year=2026"),
    ("What was Costco's net income in fiscal 2025?",
     "SELECT net_income FROM financials WHERE ticker='COST' AND fiscal_year=2025"),
    ("Which of the six retailers had the highest net income in fiscal 2025?",
     "SELECT ticker FROM financials WHERE fiscal_year=2025 ORDER BY net_income DESC LIMIT 1"),
    ("What was Target's diluted EPS in fiscal 2024 and fiscal 2025?",
     "SELECT fiscal_year, eps_diluted FROM financials WHERE ticker='TGT' AND fiscal_year IN (2024,2025) ORDER BY fiscal_year"),
    ("What was Home Depot's gross margin (gross profit divided by revenue) in fiscal 2025?",
     "SELECT gross_profit/revenue FROM financials WHERE ticker='HD' AND fiscal_year=2025"),
    ("By what percentage did Lowe's revenue change from fiscal 2024 to fiscal 2025?",
     "SELECT (a.revenue-b.revenue)/b.revenue FROM financials a JOIN financials b ON a.ticker=b.ticker "
     "WHERE a.ticker='LOW' AND a.fiscal_year=2025 AND b.fiscal_year=2024"),
    ("Rank the six retailers by fiscal 2025 revenue from largest to smallest.",
     "SELECT ticker FROM financials WHERE fiscal_year=2025 ORDER BY revenue DESC"),
    ("Which retailer had the highest operating margin in fiscal 2025?",
     "SELECT ticker FROM financials WHERE fiscal_year=2025 ORDER BY operating_income/revenue DESC LIMIT 1"),
    ("How much did Best Buy spend on capital expenditures in fiscal 2025?",
     "SELECT capex FROM financials WHERE ticker='BBY' AND fiscal_year=2025"),
    ("What was Walmart's inventory at the end of fiscal 2026?",
     "SELECT inventory FROM financials WHERE ticker='WMT' AND fiscal_year=2026"),
    ("How much did Home Depot pay in dividends in fiscal 2025?",
     "SELECT dividends_paid FROM financials WHERE ticker='HD' AND fiscal_year=2025"),
    ("What was Target's operating cash flow minus capital expenditures (free cash flow) in fiscal 2025?",
     "SELECT operating_cash_flow-capex FROM financials WHERE ticker='TGT' AND fiscal_year=2025"),
    ("Which company had the lowest net margin in fiscal 2025?",
     "SELECT ticker FROM financials WHERE fiscal_year=2025 ORDER BY net_income/revenue ASC LIMIT 1"),
    ("What was Costco's revenue growth from fiscal 2025 to fiscal 2026?",
     "SELECT (a.revenue-b.revenue)/b.revenue FROM financials a JOIN financials b ON a.ticker=b.ticker "
     "WHERE a.ticker='COST' AND a.fiscal_year=2026 AND b.fiscal_year=2025"),
    ("How much did Lowe's spend on share repurchases in fiscal 2025?",
     "SELECT share_repurchases FROM financials WHERE ticker='LOW' AND fiscal_year=2025"),
    ("What was the combined revenue of Home Depot and Lowe's in fiscal 2025?",
     "SELECT sum(revenue) FROM financials WHERE ticker IN ('HD','LOW') AND fiscal_year=2025"),
    ("In how many fiscal years since 2020 did Best Buy's revenue decline versus the prior year?",
     "SELECT count(*) FROM (SELECT revenue < lag(revenue) OVER (ORDER BY fiscal_year) AS d "
     "FROM financials WHERE ticker='BBY') WHERE d"),
    ("In the online retail data, which country had the highest total sales excluding the United Kingdom and cancellations?",
     "SELECT country FROM online_retail WHERE NOT is_cancellation AND country<>'United Kingdom' "
     "GROUP BY country ORDER BY sum(line_revenue) DESC LIMIT 1"),
    ("In the online retail data, what was total sales in November 2010, excluding cancellations?",
     "SELECT sum(line_revenue) FROM online_retail WHERE NOT is_cancellation "
     "AND invoice_date >= '2010-11-01' AND invoice_date < '2010-12-01'"),
    ("In the online retail data, how many distinct customers placed orders in 2011?",
     "SELECT count(DISTINCT customer_id) FROM online_retail WHERE NOT is_cancellation "
     "AND year(invoice_date)=2011 AND customer_id IS NOT NULL"),
    ("In the online retail data, which product description sold the most units overall (excluding cancellations)?",
     "SELECT description FROM online_retail WHERE NOT is_cancellation GROUP BY description "
     "ORDER BY sum(quantity) DESC LIMIT 1"),
    ("In the online retail data, how many invoices were cancellations?",
     "SELECT count(DISTINCT invoice) FROM online_retail WHERE is_cancellation"),
    ("In the online retail data, which month had the highest sales, excluding cancellations?",
     "SELECT strftime(date_trunc('month', invoice_date), '%Y-%m') FROM online_retail WHERE NOT is_cancellation "
     "GROUP BY 1 ORDER BY sum(line_revenue) DESC LIMIT 1"),
    ("In the online retail data, what was the average order value in 2011 (sales divided by distinct invoices, excluding cancellations)?",
     "SELECT sum(line_revenue)/count(DISTINCT invoice) FROM online_retail WHERE NOT is_cancellation "
     "AND year(invoice_date)=2011"),
    ("In the online retail data, how many countries had at least 1000 invoice lines?",
     "SELECT count(*) FROM (SELECT country FROM online_retail GROUP BY country HAVING count(*) >= 1000)"),
]

# Hybrid questions need a number from the warehouse and an explanation from the 10-K text.
# (question, reference SQL, ticker, fiscal_year, item that holds the explanation)
HYBRID_QUESTIONS = [
    ("What was Target's revenue in fiscal 2025, and what does Target say drove its comparable sales change?",
     "SELECT revenue FROM financials WHERE ticker='TGT' AND fiscal_year=2025", "TGT", 2025, "7"),
    ("What was Walmart's net income in fiscal 2026, and how does Walmart describe its eCommerce growth?",
     "SELECT net_income FROM financials WHERE ticker='WMT' AND fiscal_year=2026", "WMT", 2026, "7"),
    ("How much did Costco spend on capital expenditures in fiscal 2026, and what does it plan to spend on next year?",
     "SELECT capex FROM financials WHERE ticker='COST' AND fiscal_year=2026", "COST", 2026, "7"),
    ("What was Home Depot's operating income in fiscal 2025, and what acquisition does the 10-K discuss?",
     "SELECT operating_income FROM financials WHERE ticker='HD' AND fiscal_year=2025", "HD", 2025, "7"),
    ("What was Best Buy's revenue in fiscal 2026, and what does Best Buy say about its comparable sales?",
     "SELECT revenue FROM financials WHERE ticker='BBY' AND fiscal_year=2026", "BBY", 2026, "7"),
    ("How much did Lowe's spend on share repurchases in fiscal 2025, and how does it describe its capital allocation?",
     "SELECT share_repurchases FROM financials WHERE ticker='LOW' AND fiscal_year=2025", "LOW", 2025, "7"),
    ("What was Target's diluted EPS in fiscal 2025, and what risks to profitability does Target highlight?",
     "SELECT eps_diluted FROM financials WHERE ticker='TGT' AND fiscal_year=2025", "TGT", 2025, "1A"),
    ("What was Walmart's inventory at the end of fiscal 2026, and how does Walmart describe inventory risk?",
     "SELECT inventory FROM financials WHERE ticker='WMT' AND fiscal_year=2026", "WMT", 2026, "1A"),
    ("What was Costco's net income in fiscal 2026, and how important are membership fees to its profitability?",
     "SELECT net_income FROM financials WHERE ticker='COST' AND fiscal_year=2026", "COST", 2026, "7"),
    ("What was Home Depot's revenue in fiscal 2024, and what did management say about the housing market?",
     "SELECT revenue FROM financials WHERE ticker='HD' AND fiscal_year=2024", "HD", 2024, "7"),
    ("What was Lowe's gross profit in fiscal 2024, and what does Lowe's say about Pro customers?",
     "SELECT gross_profit FROM financials WHERE ticker='LOW' AND fiscal_year=2024", "LOW", 2024, "7"),
    ("What did Best Buy pay in dividends in fiscal 2025, and what cybersecurity risks does it describe?",
     "SELECT dividends_paid FROM financials WHERE ticker='BBY' AND fiscal_year=2025", "BBY", 2025, "1A"),
    ("What was Walmart's capital expenditure in fiscal 2025, and where is Walmart investing?",
     "SELECT capex FROM financials WHERE ticker='WMT' AND fiscal_year=2025", "WMT", 2025, "7"),
    ("What was Target's operating income in fiscal 2024, and how does Target describe tariff risk?",
     "SELECT operating_income FROM financials WHERE ticker='TGT' AND fiscal_year=2024", "TGT", 2024, "1A"),
    ("What was Costco's revenue in fiscal 2025, and how many warehouses did it operate?",
     "SELECT revenue FROM financials WHERE ticker='COST' AND fiscal_year=2025", "COST", 2025, "1"),
]

# Questions the corpus cannot answer: the system should say so instead of guessing.
UNANSWERABLE = [
    "What was Amazon's net income in fiscal 2025?",
    "What was Target's revenue in fiscal 2015?",
    "How much was Walmart's CEO paid in fiscal 2026?",
    "What is Best Buy's current stock price?",
    "What will Costco's revenue be in fiscal 2028?",
    "How many employees does Kroger have?",
    "What was Home Depot's revenue in its Canadian stores alone in fiscal 2025?",
    "In the online retail data, what were total sales in 2015?",
    "What is Lowe's market share in the paint category in Germany?",
    "What did Target's board discuss in its March 2026 meeting?",
]
