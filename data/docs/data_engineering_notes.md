# Data Engineering Notes

## Medallion Architecture

The medallion architecture organizes lakehouse data into bronze, silver, and gold layers. Bronze stores raw ingested data, silver stores cleaned and conformed data, and gold stores business-ready datasets used by analytics, machine learning, and reporting.

## ETL

ETL stands for extract, transform, and load. Data is extracted from source systems, transformed into a useful structure, and loaded into a target system such as a warehouse, lakehouse, or feature store.

## Partitioning

Partitioning divides large datasets into smaller physical sections, often by date or region. Good partitioning improves query performance and reduces the amount of data scanned. Poor partitioning can create too many small files or uneven data distribution.

## Delta Lake

Delta Lake adds ACID transactions, schema enforcement, time travel, and scalable metadata handling to data lakes. It is commonly used to make data lake storage more reliable for analytics and machine learning workloads.

