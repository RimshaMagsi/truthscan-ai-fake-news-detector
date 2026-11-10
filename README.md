# TRUTHSCAN AI — Fake News Detection

## Setup
1. Put `Fake.csv` and `True.csv` in the `dataset` folder.
2. Create and activate a virtual environment.
3. Install requirements.
4. Train the model with `python train_model.py`.
5. Launch the app with `streamlit run app.py`.

## Windows PowerShell commands
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

This educational model learns patterns from the ISOT dataset. It is not a live fact-checking service.
