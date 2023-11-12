
# FISA
### The implementation of pseudo value-based fair and interpretable deep survival models FIDP and FIPNAM.

<p align="center">
    <a href="#Paper">Paper</a> •
    <a href="#installation">Installation</a> •    
    <a href="#required-packages">Required Packages</a> •
    <a href="#running-experiments">Running Experiments</a> • 
    <a href="#datasets">Datasets</a> •    
    <a href="#evaluation-criteria">Evaluation Criteria</a> •
    <a href="#censoring-settings">Censoring Settings</a> • 
    <a href="#references">References</a> •	
    <a href="#how-to-cite">How to Cite</a> •    
    <a href="#Contact">Contact</a> •
</p>


## Paper
For more details, see full paper [Fair and Interpretable Models for Survival Analysis](https://dl.acm.org/doi/10.1145/3534678.3539259).

## Installation
### From source
Download a local copy of FISA_KDD22 and install from the directory:

	git clone https://github.com/umbc-sanjaylab/FISA_KDD22.git
	cd FISA_KDD22
	pip install .

### Configure the environement

	conda env create --name FISA
	conda activate FISA

## Required Packages
* PyTorch
* pycox
* scikit-survival 
* scikit-learn
* xlwt
* numpy
* pandas

### Install the packages
pip install -r requirements.txt


## Running Experiments
* To run the FIDP model on FLChain dataset, run the python script as 

		python Experiment.py -i ./FISA_KDD22/Data/FLC/flchain.csv -p ./FISA_KDD22 -m "FIDP" -d  "FLChain"  -b 32 -lr 0.01 -e 100

* To run the FIPNAM model on FLChain dataset, run the python script as 

		python Experiment.py -i ./FISA_KDD22/Data/FLC/flchain.csv -p ./FISA_KDD22 -m "FIPNAM" -d  "FLChain"  -b 32 -lr 0.01 -e 100
	
i: csv Data File location, 
p: Name of the Directory, 
m: Model Name, 
d: Dataset Name, 
b: Batch Size, 
lr: Learning Rate, 
e: Number of Epochs	

## Datasets 
Experiments are conducted on the follwing survival datasets.

<table>
    <tr>
        <th>Dataset</th>
        <th>Size</th>
        <th>Description</th>
        <th>Data source</th>
    </tr>
    <tr>
        <td>FLChain</td>
        <td>6,521</td>
        <td>
        The Serum Free Light Chain (FLCHAIN) dataset is collected from a study of the relationship between serum free light chain (FLC) and mortality of the Olmsted County residents aged 50 years or more.  See <a href="#references">[1]</a> for details.
        </td>
        <td><a href="https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.datasets.load_flchain.html">source</a>
    </tr>
    <tr>
        <td>SUPPORT</td>
        <td>8,950</td>
        <td>
        Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT).
        See <a href="#references">[2]</a> for details.
        </td>
        <td><a href="https://github.com/autonlab/auton-survival/tree/master/dsm/datasets">source</a>
    </tr>
    <tr>
        <td>SEER</td>
        <td>25,319</td>
        <td>
        The Surveillance, Epidemiology, and End Results (SEER) Program of National Cancer Institute provides information on the survival attributes of oncology patients in the United States.
        </td>
        <td><a href="https://seer.cancer.gov/">source</a>
    </tr>		
</table>

## Evaluation Criteria 
The following evaluation metrics are used in the paper.

<table>
    <tr>
        <th>Metric</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>concordance_td</td>
        <td>
        The time-dependent concordance index evaluated at the event times <a href="#references">[3]</a>.
        </td>
    </tr>
    <tr>
        <td>integrated_brier_score</td>
        <td>
        The integrated IPCW Brier score. Numerical integration of the Brier Scores <a href="#references">[4]</a>.
        </td>
    </tr>
    <tr>
        <td>cumulative_dynamic_auc</td>
        <td>
        Estimator of cumulative/dynamic AUC for right-censored time-to-event data. <a href="#references">[5]</a>.
        </td>
    </tr>	
</table>

## Censoring Settings 
To obtain the datasets with various censoring settings (**Uncensored**, **Increment All**, **Increment Majority**, **Increment Minority**, **Induced Majority**, **Induced Minority**), run **Different_censoring_settings.ipynb**.

## References
\[1\] Dispenzieri, Angela, et al. "Use of nonclonal serum immunoglobulin free light chains to predict overall survival in the general population." Mayo Clinic Proceedings. Vol. 87. No. 6. Elsevier, 2012.\[[paper](https://pubmed.ncbi.nlm.nih.gov/22677072/)\]

\[2\] Knaus, William A., et al. "The SUPPORT prognostic model: Objective estimates of survival for seriously ill hospitalized adults." Annals of internal medicine 122.3 (1995): 191-203. \[[paper](https://pubmed.ncbi.nlm.nih.gov/7810938/)\]

\[3\] Antolini, Laura, Patrizia Boracchi, and Elia Biganzoli. "A time‐dependent discrimination index for survival data." Statistics in medicine 24.24 (2005): 3927-3944. \[[paper](https://doi.org/10.1002/sim.2427)\]
  
\[4\] Graf, Erika, et al. "Assessment and comparison of prognostic classification schemes for survival data." Statistics in medicine 18.17‐18 (1999): 2529-2545. \[[paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0258%2819990915/30%2918%3A17/18%3C2529%3A%3AAID-SIM274%3E3.0.CO%3B2-5)\]
 
\[5\] Uno, Hajime, et al. "Evaluating prediction rules for t-year survivors with censored regression models." Journal of the American Statistical Association 102.478 (2007): 527-537. \[[paper](https://www.tandfonline.com/doi/abs/10.1198/016214507000000149)\] 

## How to Cite

	@inproceedings{rahman2022fair,
	  title={Fair and Interpretable Models for Survival Analysis},  
	  author={Rahman, Md Mahmudur and Purushotham, Sanjay},  
	  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},  
	  pages={1452--1462},  
	  year={2022}  
	}
  
## Contact
* Md Mahmudur Rahman (mrahman6@umbc.edu)
* Sanjay Purushotham (psanjay@umbc.edu)
