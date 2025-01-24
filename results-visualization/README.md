# Statistical Analysis and Data Visualization
**Authors:** *Qinglei Shi*, *Yangfan Sun*

## Project Overview

This project focuses on data analysis related to feature combination, calculating various metrics, and generating ROC curves with confidence intervals. The codebase consists of several Python scripts that perform different tasks in the analysis pipeline.

## Directory Structure

- `combine_features/`: Contains scripts that predict disease and generate ROC curves and statistics results by some feature combination methods.
    - `combine_features/combine_1/`: Contains scripts related to feature selection and combination by AUC.
    
    - `combine_features/combine_stepwise/`: Contains scripts related to stepwise feature combination methods.
    
    - `combine_features/make_tables/`: Contains scripts for generating tables of analysis results.

- `rocs_seperated/`: Contains scripts related to generate ROC Curves of 'Mean' and 'Median' seperatedly.

- `data/`:
    - `mask_group_differential_Haoran/`: Contains intermittent exotropia clinical data. `jianwai/` is the experimental data and `zhengwei/` is the control data.
    - `mean_median_results/`: Contains some statistical results from previous analysis, including AUCs and CIs.
    - `stat_results`: Contains some statistical results, including P-values.

- `all_results`: Contains all the results from the Python scripts, including ROC curves and tables.

## Main Scripts and Their Functions

### `rocs_main.py`
The purpose of this script is to generate roc curves of `mean` and `median` separately to see how each group of `mean` and `median` reflect this disease

### `combine_main.py`
The main purpose of this script is to process experimental and control data, generate probability data, plot ROC curves and calculate confidence intervals. Different combinations of data are processed and analyzed through nested loops traversing different `means`/`medians` and thresholds (AUC).

### `stepwise_combine_main.py`
The main purpose of this script is to process experimental and control data, generate probability data, plot ROC curves and calculate confidence intervals. Different combinations of data are processed and analyzed through nested loops traversing different `means`/`medians` and different combination methods (`forward`, `backward`, `bidirection`).

### `auc_tb_main.py`
The main function of this scripts is to process the experimental data and the control group data, calculate the AUC related results under different feature selection thresholds (AUCs), and save the final results as CSV files.

### `step_tb_main.py`
The main function of this scripts is to process the experimental data and the control group data, calculate the AUC related results under different combination methods (`forward`, `backward`, `bidirection`), and save the final results as CSV files.

## Dependencies
- Python 3.x
- Libraries: `pandas`, `numpy`, `sklearn`, `matplotlib`, `statsmodels`, `glob`, `os`

## Usage
To run the analysis, simply execute the main scripts (`rocs_main.py`, `combine_main.py`, `stepwise_combine_main.py`, `auc_tb_main.py`, or `step_tb_main.py`) using Python. Make sure all the required data files are present in the specified directories.

```bash
python analysis_codes/roc_seperated/rocs_main.py
python analysis_codes/combine_features/combine_1/combine_main.py
python analysis_codes/combine_features/combine_stepwise/stepwise_combine_main.py
python analysis_codes/combine_features/make_tables/auc_tb_main.py
python analysis_codes/combine_features/make_tables/step_tb_main.py
```

## Output
The analysis produces the following outputs:
- ROC curve plots with confidence intervals, saved as JPEG files in the `all_results/ROCs_seperated`, `all_results/combine_rocs` and `all_results/stepwise_rocs` directories.
- Tables of analysis results, saved as CSV files in the `all_results` directory.

## Future Improvements
- Add more detailed error handling and logging to improve the robustness of the code.
- Optimize the code for performance, especially in the loops that perform bootstrapping.
- Provide more flexibility in the input parameters, such as allowing users to specify different feature types and thresholds from the command line.