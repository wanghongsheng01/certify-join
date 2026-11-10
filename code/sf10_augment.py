"""Add the business-key fact table sales_bk (SCD joins on item / store business ids, as in the SF1 instance: a 10%
sample of store_sales with item id, store id and sale date) to the TPC-DS SF10 instance on the external disk."""
import duckdb
c = duckdb.connect("<DATA_DIR>/data/tpcds_sf10.duckdb")
c.execute("DROP TABLE IF EXISTS sales_bk")
c.execute("""CREATE TABLE sales_bk AS
  SELECT ss_ticket_number AS sb_ticket_number, i_item_id AS sb_item_id, s_store_id AS sb_store_id, d_date AS sb_sold_date,
         ss_net_paid AS sb_net_paid
  FROM (SELECT * FROM store_sales USING SAMPLE 10 PERCENT (bernoulli, 7)) ss
  JOIN item ON ss_item_sk = i_item_sk JOIN store ON ss_store_sk = s_store_sk JOIN date_dim ON ss_sold_date_sk = d_date_sk""")
c.execute("CHECKPOINT")
print(c.execute("SELECT count(*) FROM sales_bk").fetchone()); c.close()
