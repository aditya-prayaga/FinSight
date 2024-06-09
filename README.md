# FinSight
Adaptive Stock Prediction and Monitoring Pipeline

### Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Data Pipeline Workflow](#data-pipeline-workflow)
7. [Testing](#testing)
8. [Logging and Tracking](#logging-and-tracking)
9. [Data Version Control](#data-version-control)
10. [Pipeline Optimization](#pipeline-optimization)
11. [Schema and Statistics Generation](#schema-and-statistics-generation)
12. [Anomalies Detection and Alerts](#anomalies-detection-and-alerts)
13. [Conclusion](#conclusion)

### Introduction
This project demonstrates a comprehensive pipeline for processing stock data and making a prediction. It includes data preprocessing, testing, workflow orchestration with Apache Airflow, data versioning with DVC, schema generation, and anomaly detection. The project is documented to ensure replication on other machines.

### Project Structure
This Project will be designed to be robust in both development and production setup. We Created 2 branches

1. ```dev``` for local development version with local storage & deployment
2. ```main``` for production ready version with Google storage & deployment

```
.
├── DAGs
│   └── FinSight
│       ├── __init__.py
│       ├── finsight_pipeline_functions.py
│       └── finsight_pipeline_taks.py
├── LICENSE
├── README.md
├── data
├── docs
├── model
├── scoping
│   └── Finsight-scoping-doc-final.pdf
├── tests
└── visualizations
    └── raw-data-viz.png
```

### Prerequisites
- Python 3.7+ (Packages like tensorflow, sklearn, etc.)
- Apache Airflow 2.5+
- Google Cloud Services
- Google Cloud SDK (for GCS integration)

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/aditya-prayaga/FinSight.git
   cd FinSight
    ```
    **Note:** 
    1. If the user trying to install doesn't have airflow setup/docker setup done, please follow instructions in this [video](https://www.youtube.com/watch?v=exFSeGUbn4Q&t=511s&pp=ygUTcmFtaW4gYWlyZmxvdyBzZXR1cA%3D%3D)
    2. Please git check out ```dev``` branch to find additional files like ```docker-compose.yaml```, logs etc which help us in deploying the airflow and orchestrate.

2. dsd



## Project Information
## Installation Instructions
## Usage Guidelines
