# Smartphone Rating Predictor

Predicting smartphone user ratings from specifications using machine learning.

Final project of Chatpol Phongyen (st127020) and Nyi Htut Oo (st126005) for the course
Computer Programming for Data Science and Artificial Intelligence, Asian Institute of Technology (AIT), November 2026.

**Live app (AWS EC2):** http://ec2-3-90-222-164.compute-1.amazonaws.com:8080/
(online only while the AWS lab session and the instance are running; the address changes after every restart)

## What the project does

Most buyers choose a phone by its average user rating, but a new phone has few or no reviews yet.
This project predicts the average user rating (out of 5) from the phone's specifications and price.

- Three real Kaggle datasets are screened, cleaned and matched (1,597 phones released 2022–2026).
- A baseline, linear regression, random forest and a tuned XGBoost model are compared under four protocols:
  a Kaggle-style random split, grouped five-fold cross-validation, leave-one-brand-out and a future-year split.
- Review volume is used to measure how reliable each rating is, and to weight the training labels.
- The final XGBoost model is delivered as a Streamlit web app that explains each prediction with SHAP.

### Main results (XGBoost, mean absolute error in rating points)

| Protocol | XGBoost | Baseline |
|---|---|---|
| Grouped five-fold CV (new phones of known brands) | 0.153 | 0.203 |
| Leave-one-brand-out (unseen brands) | 0.177 | 0.207 |
| Future year (trained on ≤2024, tested on 2025–26) | 0.205 | 0.256 |
| Random split (Kaggle style, for comparison only) | 0.146 | 0.204 |

## Requirements

- Python 3.13 (Python 3.12 or newer is needed for XGBoost 3.4.1 and SciPy 1.18)
- The libraries in `requirements.txt` (pandas 3.0.1, NumPy 2.4.3, scikit-learn 1.9.0, XGBoost 3.4.1, SciPy 1.18.1,
  statsmodels 0.15.0, Streamlit 1.67.0 and others). Use these exact versions, because the saved model loads
  correctly only with the versions it was trained with.

## Setup

1. Download the code:

   ```
   git clone https://github.com/nyihtutoo/Smartphone_Rating_Predictor.git
   cd Smartphone_Rating_Predictor
   ```

   (Or click **Code → Download ZIP** on GitHub and unzip it.)

2. Create a virtual environment and install the libraries (once):

   Windows (PowerShell or Command Prompt):
   ```
   python -m venv venv
   venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```

   macOS / Linux:
   ```
   python3 -m venv venv
   source venv/bin/activate
   python -m pip install -r requirements.txt
   ```

## How to run

All commands are run inside the `Smartphone_Rating_Predictor` folder with the virtual environment active.

### 1. Start the web app

```
python -m streamlit run app/app.py
```

Then open http://localhost:8080 in your web browser. The port and colours come from `.streamlit/config.toml`,
so start the app from this folder. Pick a real phone under "Start from a real phone", or enter the
specifications yourself. For example, the Tecno Spark 9 3GB RAM gives a predicted rating of 4.08.

### 2. Reproduce all results (optional)

```
python run_pipeline.py
```

or open `Final_Pipeline.ipynb` in Jupyter and choose **Kernel → Restart & Run All**.
This takes about 6–7 minutes. It writes every table to `results/`, every figure to `figures/`
and the final model to `models/`. The random seed is 42 everywhere, so the results are the same on every run
(only training and prediction times change).

If the saved model cannot be loaded on a new computer, run `python retrain_model.py` to train it again
(about 10 seconds).

## Folder contents

| Folder / file | What it is |
|---|---|
| `app/app.py` | The web app (Smartphone Rating Predictor) |
| `.streamlit/config.toml` | Colours and fonts of the app, and the server settings (address 0.0.0.0, port 8080) |
| `src/prep.py` | Loading, parsing, phone families, review-count parser and two-step matching of the datasets |
| `src/models.py` | Feature sets, pre-processing pipeline, baseline and models |
| `src/evaluation.py` | The four protocols, metrics, bootstrap intervals and paired tests |
| `src/ordinal.py` | Three-level (Low / Medium / High) task and ordinal logistic regression |
| `src/pipeline.py` | All steps of the pipeline; writes every table and figure |
| `src/explain.py` | Prediction, SHAP explanation and similar phones for the app |
| `src/report_pdf.py` | One-page PDF summary that the app can download |
| `Final_Pipeline.ipynb` | Runs the whole pipeline and shows every table and figure |
| `run_pipeline.py` | The same pipeline from the command line |
| `retrain_model.py` | Trains the final model again, only if the saved model cannot be loaded |
| `data/` | The three datasets from Kaggle, unchanged |
| `models/` | The final model (`final_model_xgb.json` + `final_model_pre.joblib`; `final_model.joblib` as a backup), its settings and the reference phones used by the app |
| `results/` | All result tables (CSV), including the manual check of 100 Dataset 3 matches |
| `figures/` | All figures (PNG) |
| `requirements.txt` | Library versions |

## Deployment on AWS EC2

The app is deployed on an Amazon EC2 instance (AWS Academy Learner Lab): t3.micro (2 vCPU, 1 GiB memory),
Amazon Linux 2023, 8 GiB gp3 storage. The security group allows SSH (port 22) and the app (port 8080) from anywhere.

| Step | Where | Command |
|---|---|---|
| 1. Protect the key file | Laptop, PowerShell | `icacls .\<key>.pem /inheritance:r`<br>`icacls .\<key>.pem /remove *S-1-5-32-545 *S-1-5-11 *S-1-1-0`<br>`icacls .\<key>.pem /grant:r "$($env:USERNAME):(R)"` |
| 2. Install Python 3.13 | EC2 terminal | `sudo dnf install python3.13 python3.13-pip -y` |
| 3. Copy the project | Laptop, PowerShell | `scp -r -i .\<key>.pem .\Smartphone_Rating_Predictor ec2-user@<public DNS>:~/` |
| 4. Install the libraries | EC2 terminal | `cd ~/Smartphone_Rating_Predictor`<br>`python3.13 -m pip install xgboost-cpu==3.4.1`<br>`python3.13 -m pip install pandas==3.0.1 numpy==2.4.3 scikit-learn==1.9.0 scipy==1.18.1 matplotlib==3.11.1 statsmodels==0.15.0 joblib==1.6.0 streamlit==1.67.0 altair` |
| 5. Start the app | EC2 terminal | `nohup python3.13 -m streamlit run app/app.py &` |
| 6. Open the app | Browser | `http://<public DNS>:8080` |
| 7. Stop the app | EC2 terminal | `pkill -f streamlit` |

Notes:
- Replace `<key>` with the name of the key pair file and `<public DNS>` with the public DNS shown on the EC2 instance page.
- `xgboost-cpu` is the same XGBoost 3.4.1 without the GPU files, so it is much smaller to download. The saved model loads the same way.
- Jupyter is not installed on the server because the app does not need it.
- After the instance is stopped and started again, check the new public DNS and run step 5 again.

## Datasets

1. Khadka, S. (2026). *Mobile Phones Specs & Prices Dataset (2008–2026)* (Version 1) [Data set]. Kaggle. CC0.
   https://www.kaggle.com/datasets/suresh2837/mobile-phones-specs-and-prices-dataset-20082026
2. Vashisth, R. (2025). *Mobile price and specs [2025]* (Version 1) [Data set]. Kaggle. CC0.
   https://www.kaggle.com/datasets/vashisthrahul13/mobile-price-and-specs-2025
3. Dab, R. (2025). *Global Smartphone Dataset (2012–2025)* (Version 2) [Data set]. Kaggle. Apache 2.0.
   https://www.kaggle.com/datasets/rajibdab/global-smartphone-database-2025
