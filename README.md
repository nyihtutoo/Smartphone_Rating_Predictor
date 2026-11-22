# Neat 2 — Final Project: Predicting Smartphone User Ratings

Chatpol Phongyen (st127020), Nyi Htut Oo (st126005) — Computer Programming for Data Science and AI, AIT.

## Folder contents

| Folder / file | What it is |
|---|---|
| `data/` | The three datasets (copied from Kaggle, unchanged) |
| `src/` | The pipeline code: `prep.py` (cleaning, phone families, matching), `models.py` (features and models), `evaluation.py` (protocols, metrics, bootstrap), `ordinal.py` (3-level task), `pipeline.py` (all steps), `explain.py` (prediction, SHAP and similar phones for the app), `report_pdf.py` (one-page PDF summary for the app) |
| `Final_Pipeline.ipynb` | Runs the whole pipeline and shows every table and figure |
| `run_pipeline.py` | Same as the notebook, from the command line |
| `results/` | All result tables (CSV), including the manual check of 100 Dataset 3 matches (`table_d3_match_audit.csv`) |
| `figures/` | All figures (PNG) |
| `models/` | The final model (`final_model_xgb.json` + `final_model_pre.joblib`, which load the same on every computer; `final_model.joblib` as a backup), its settings and the reference phones used by the app |
| `app/app.py` | The demo app (Smartphone Rating Predictor) |
| `.streamlit/config.toml` | Colours and fonts of the app, and the server settings (address 0.0.0.0, port 8080) |
| `retrain_model.py` | Trains the final model again on this computer, only if the saved model cannot be loaded |
| `requirements.txt` | Library versions |

## How to run

1. Install Python 3.13 and the libraries (once), ideally in a virtual environment:
   `python -m venv venv`, `venv\Scripts\activate` (Windows), then `pip install -r requirements.txt`
2. Run the pipeline (about 7 minutes):
   open `Final_Pipeline.ipynb` and choose **Kernel → Restart & Run All**, or run `python run_pipeline.py`.
3. Start the demo app from this folder (so the colours in `.streamlit/` are used):
   `python -m streamlit run app/app.py`, then open http://localhost:8080 in your web browser.

Random seed = 42 everywhere. All results reproduce exactly on the same system; training and prediction times
change from run to run.

## The app

Pick a real phone or enter the specifications (price in US$, converted at about ₹94 per US$). The app shows the
predicted rating, its level (Low / Medium / High), the typical error for the situation (known brand, new brand or a
2025–26 phone), a SHAP chart of the reasons, five similar real phones, and a **Download summary (PDF)** button.

## Deployment on AWS EC2

The app runs on an Amazon EC2 instance (AWS Academy Learner Lab): t3.micro, Amazon Linux 2023, 8 GiB gp3.
The security group allows SSH (port 22) and the app (port 8080) from anywhere.

Live app (only while the lab session and the instance are running; the address changes after every restart):
http://ec2-3-90-222-164.compute-1.amazonaws.com:8080

| Step | Where | Command |
|---|---|---|
| 1. Protect the key file | Laptop, PowerShell | `icacls .\<key>.pem /inheritance:r` and `icacls .\<key>.pem /grant:r "$($env:USERNAME):(R)"` |
| 2. Install Python 3.13 | EC2 terminal | `sudo dnf install python3.13 python3.13-pip -y` |
| 3. Copy the project | Laptop, PowerShell | `scp -r -i .\<key>.pem .\Smartphone_Rating_Predictor ec2-user@<public DNS>:~/` |
| 4. Install the libraries | EC2 terminal | `cd ~/Smartphone_Rating_Predictor`, then `python3.13 -m pip install xgboost-cpu==3.4.1`, then `python3.13 -m pip install pandas==3.0.1 numpy==2.4.3 scikit-learn==1.9.0 scipy==1.18.1 matplotlib==3.11.1 statsmodels==0.15.0 joblib==1.6.0 streamlit==1.67.0 altair` |
| 5. Start the app | EC2 terminal | `nohup python3.13 -m streamlit run app/app.py &` |
| 6. Open the app | Browser | `http://<public DNS>:8080` |
| 7. Stop the app | EC2 terminal | `pkill -f streamlit` |

Notes:
- `xgboost-cpu` is the same XGBoost 3.4.1 without the GPU files, so it is much smaller to download. The saved model loads the same way.
- Jupyter is not installed on the server because the app does not need it.
- The port and address come from `.streamlit/config.toml`, so the app must be started from inside the project folder.
- After the instance is stopped and started again, check the new public DNS and run step 5 again.

## Datasets

1. Khadka, S. (2026). *Mobile Phones Specs & Prices Dataset (2008–2026)* (Version 1) [Data set]. Kaggle. CC0.
2. Vashisth, R. (2025). *Mobile price and specs [2025]* (Version 1) [Data set]. Kaggle. CC0.
3. Dab, R. (2025). *Global Smartphone Dataset (2012–2025)* (Version 2) [Data set]. Kaggle. Apache 2.0.
   (Kaggle address: rajibdab/global-smartphone-database-2025; the dataset was earlier titled "Global Smartphone Database 2025".)
