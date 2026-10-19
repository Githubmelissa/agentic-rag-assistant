"""Prompt templates, kept in one place so they can be reviewed and versioned together."""

NOT_FOUND = "I could not find this in the documents."

DOC_QA_SYSTEM = f"""You answer questions about company reports using ONLY the context passages \
provided. Each passage starts with a label such as [kering_urd_2025 p.47].

Rules:
1. Use only facts stated in the context. Never use outside knowledge, and never guess.
2. After every sentence that states a fact, cite the passage it comes from by copying its \
label exactly, e.g. [kering_urd_2025 p.47].
3. Copy numbers, units and years exactly as written in the context.
4. If the context does not contain the answer, reply with exactly this sentence and \
nothing else: {NOT_FOUND}
5. Be concise: at most 4 sentences."""

DOC_QA_USER = """Context:
{context}

Question: {question}

Answer (with citations):"""


SQL_SYSTEM = """You write one SQLite SELECT query that answers a question about a retail \
sales database. Reply with the SQL only, in a ```sql code block, with no explanation.

{schema}

Examples:
Question: What was the total revenue in 2011?
```sql
SELECT ROUND(SUM(line_total), 2) AS revenue FROM sales_clean
WHERE invoice_date >= '2011-01-01' AND invoice_date < '2012-01-01'
```
Question: Which 5 countries outside the United Kingdom had the most orders?
```sql
SELECT country, COUNT(DISTINCT invoice) AS orders FROM sales_clean
WHERE country <> 'United Kingdom' GROUP BY country ORDER BY orders DESC LIMIT 5
```
Question: How many orders were placed per month in 2010?
```sql
SELECT strftime('%Y-%m', invoice_date) AS month, COUNT(DISTINCT invoice) AS orders
FROM sales_clean WHERE invoice_date >= '2010-01-01' AND invoice_date < '2011-01-01'
GROUP BY month ORDER BY month
```"""

SQL_USER = "Question: {question}"

SQL_RETRY_USER = """Question: {question}

Your previous query:
```sql
{sql}
```
failed with this error: {error}
Write a corrected query."""

# Used as the "error" in SQL_RETRY_USER when a query ran but returned nothing useful.
SQL_EMPTY_RESULT = (
    "no error, but the query returned no rows or only NULL values. Check that you used "
    "the right table or view (see the rules), that the filters are not too strict, and "
    "that text values are spelled exactly as stored (text comparisons are case-sensitive)."
)

SQL_ANSWER_SYSTEM = """You explain the result of a database query in one or two plain \
sentences. Use only the numbers in the result, copied exactly. Mention the unit (GBP, \
orders, customers...) when it is clear. If the result is empty, say that no matching data \
was found."""

SQL_ANSWER_USER = """Question: {question}
SQL: {sql}
Result (columns: {columns}; {n_rows} rows{truncated}):
{rows}

Answer:"""


ROUTES = ("documents", "sales_data", "both", "out_of_scope")

ROUTER_SYSTEM = """Classify a user question into exactly one category. Reply with the \
category name only.

Categories:
- documents: about the 2025 annual or sustainability reports of Kering, Hermès or LVMH \
(revenue, strategy, employees, governance, emissions, brands...).
- sales_data: about the transactions of a UK online gift shop in 2009-2011 (sales, orders, \
products, customers, countries, cancellations).
- both: needs facts from the company reports AND from the gift-shop sales data.
- out_of_scope: anything else (general knowledge, other companies, advice, chit-chat).

Examples:
What was Hermès' revenue in 2025? -> documents
Which country bought the most from the gift shop in 2011? -> sales_data
Compare Kering's 2025 revenue with the gift shop's total revenue in 2011. -> both
What is the capital of Australia? -> out_of_scope"""

ROUTER_USER = "Question: {question}\nCategory:"

OUT_OF_SCOPE_ANSWER = (
    "I can only answer questions about two sources: the 2025 reports of Kering, Hermès and "
    "LVMH (with cited pages), and the 2009-2011 sales data of a UK online gift shop. "
    "Your question does not seem to be about either of them."
)

SPLIT_SYSTEM = """Two sources are available: (1) the 2025 reports of Kering, Hermès and \
LVMH, and (2) the 2009-2011 sales database of a UK online gift shop. Split the question \
into one self-contained sub-question per source, using ONLY words and numbers from the \
question itself. Never invent a topic, a company or a year that the question does not \
mention. If the question needs only one source, write NONE for the other one. Reply with \
exactly two lines:
REPORTS: <question for the company reports, or NONE>
SALES: <question for the sales database, or NONE>

Example 1:
Question: Compare Hermès' 2025 revenue with the gift shop's revenue in 2010.
REPORTS: What was Hermès' revenue in 2025?
SALES: What was the total revenue in 2010?

Example 2:
Question: Who is the Chief Executive Officer of Kering?
REPORTS: Who is the Chief Executive Officer of Kering?
SALES: NONE"""

SPLIT_USER = "Question: {question}"

BOTH_TEMPLATE = """From the company reports: {doc_answer}

From the gift-shop sales data: {sql_answer}"""
