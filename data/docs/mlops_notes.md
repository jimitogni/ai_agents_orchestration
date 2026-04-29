# MLOps Notes

## MLflow

MLflow is an open-source platform for tracking experiments, packaging models, and managing model versions. Teams use it to compare parameters, metrics, artifacts, and trained models across runs.

## Airflow

Apache Airflow schedules and orchestrates workflows as directed acyclic graphs. In MLOps, Airflow can coordinate data extraction, feature generation, model training, batch scoring, and retraining jobs.

## Model Monitoring

Model monitoring tracks model quality and operational health after deployment. Useful signals include prediction distributions, latency, error rates, missing values, feature drift, concept drift, and business outcome metrics.

## Drift and Retraining

Data drift happens when production input data changes compared with training data. Concept drift happens when the relationship between inputs and the target changes. Retraining should be triggered by meaningful drift, performance degradation, scheduled refresh windows, or newly labeled data.

