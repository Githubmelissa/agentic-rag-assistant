# Database: retail sales (SQLite)

Transactions of a UK-based online gift shop, 2009-12-01 to 2011-12-09. Amounts in pounds
sterling (GBP). Source: UCI Online Retail II.

## View `sales_clean` (use this for sales questions)

Only real product sales: no cancellations, no postage/fees/adjustments, quantity > 0,
unit_price > 0. Same columns as `sales`.

## Table `sales` (all lines, including cancellations and non-product lines)

| column | type | meaning |
|---|---|---|
| invoice | TEXT | invoice number; starts with 'C' for a cancellation |
| stock_code | TEXT | product code |
| description | TEXT | product name, upper case (may be NULL) |
| quantity | INTEGER | units; negative for cancellations |
| invoice_date | TEXT | 'YYYY-MM-DD HH:MM:SS' |
| unit_price | REAL | price per unit in GBP |
| line_total | REAL | quantity * unit_price, in GBP |
| customer_id | INTEGER | customer number; NULL when unknown |
| country | TEXT | customer country, e.g. 'United Kingdom', 'France', 'EIRE' (Ireland) |
| is_cancelled | INTEGER | 1 for cancellation lines |
| is_product | INTEGER | 0 for postage, fees, manual adjustments, gift vouchers |

## Rules

- Revenue / sales = SUM(line_total). An order = a distinct invoice: count orders with
  COUNT(DISTINCT invoice), never COUNT(*) (one invoice has several lines). This also
  applies to cancelled orders.
- A product is identified by stock_code (the same product can have several descriptions):
  GROUP BY stock_code and show MAX(description) as its name.
- `sales_clean` gives GROSS sales. Some orders were cancelled later; for NET sales or
  quantities, add the cancellation lines of `sales` (they have negative quantity and
  line_total) to the gross figures.
- Use `sales_clean` unless the question is about cancellations, returns, postage or fees.
- Filter dates with string comparison, e.g. invoice_date >= '2011-01-01' AND
  invoice_date < '2012-01-01'; group by month with strftime('%Y-%m', invoice_date).
- Ireland is stored as 'EIRE'. Text values are case-sensitive; use UPPER() or LIKE for
  product names.
- Exclude NULL customer_id when counting customers.
