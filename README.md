# CTO-HF In-hospital Mortality Risk Calculator

Streamlit web deployment for the final Gradient boosting model predicting in-hospital mortality among patients with chronic total occlusion and heart failure.

## Files

- `app.py`: Streamlit web application.
- `gradient_boosting_cto_hf.joblib`: final fitted Gradient boosting pipeline.
- `model_metadata.json`: predictors, threshold and validation performance.
- `cto_hf_prediction_template.csv`: template for batch prediction.
- `requirements.txt`: Python dependencies.

## Public deployment

The application is configured for deployment on Streamlit Community Cloud.
The public URL will be recorded here after cloud deployment is completed.

No patient-level development or validation data are included in this
repository.

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

On the configured Windows workstation, double-click
`启动网页计算器.bat` and open `http://127.0.0.1:8501`.

This is a local address rather than a public domain. The page is available
only while the Streamlit process is running on the workstation. A permanent
public URL requires deployment to an external hosting service.

## Required Predictors

- `Hb`
- `Crea`
- `NT_ProBNP`
- `NYHA`
- `Vascular_recanalization`

`DEATH` is not used as an input for deployment.

## Notes

The fixed classification threshold was selected in the training set and then applied to the internal and external validation cohorts. This web calculator is intended for research presentation and manuscript demonstration.
