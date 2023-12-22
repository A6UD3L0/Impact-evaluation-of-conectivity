# Impact Evaluation of Connectivity on Test Scores

## Overview

This repository contains scripts for analyzing the impact of internet connectivity on standardized test scores (Saber 11) in rural departments. The analysis uses data from ICFES spanning from 2004 to 2021. The methodology employs an event study approach on the general test while also calculating heterogeneous effects. The integration between Python and Stata is a novel feature of this project.

## Scripts

### ReadDFEstimateITT.py (Python)

#### Description

The **ReadDFEstimateITT.py** script utilizes both Stata and Python capabilities to read data frames for various years. Recursive loops and exceptions automate the process, assigning treatment effects and creating a panel data.

#### Key Steps

The script performs the following key steps:

1. **Importing Libraries:** The necessary libraries and packages are imported, including pandas, numpy, Stata-related libraries, and others.

2. **ICFES Data Import:** ICFES data from multiple files for different years is imported and processed. Columns are standardized, and missing data is handled.

3. **Treatment Data Preparation:** Data related to the treatment (connectivity) is read from CSV files containing information about the Proyecto Nacional de Fibra Óptica (PNFO) and Conectividad de Alta Velocidad (PCAV). The data is cleaned, and a binary treatment variable is created.

4. **Data Merging and Filtering:** The ICFES data is merged with treatment data based on municipality codes. Filtering is performed to include only relevant observations.

5. **Data Transformation:** Selected columns are transformed, and logarithms of test scores are calculated.

6. **Grouping and Aggregation:** The data is grouped by gender, socio-economic stratum, and municipality. Average test scores are calculated for each group.

7. **Data Export:** Processed data is exported to CSV files for further analysis.


### ReadDFEstimateITT.py (Stata)

#### Description

The **ReadDFEstimateITT.py** script includes code for estimating treatment effects using different methodologies, such as DID (Borusyak et al., 2021), CSDID (Callaway and Sant'Anna, 2020), eventstudyinteract (Sun and Abraham, 2020), and TWFE. The script also handles data cleaning, merging, and export to Stata datasets.

#### Key Steps

- Data cleaning and preprocessing for ICFES datasets (2008-2021).
- Implementation of various event study approaches for impact estimation.
- Storing estimates for later use and generating combined plots.
- Descriptive statistics and tabulations.


## Project Notes

- This project represents an exploration into novel integration between econometric tools such as Stata and data science tools like Python, making it one of the most powerful integrations for data scientists with a background in economics. It also incorporates state-of-the-art estimations for treatment effects.
- For additional details, refer to the script comments and the context provided for each 


section.
