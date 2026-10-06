"""Run `python train.py` to train all models and save the best one."""
import ml

df = ml.load_data()
b = ml.train(df)
print(f"Trained on {b['n_train']} rows, tested on {b['n_test']} rows\n")
for name, r in b["results"].items():
    print(f"{name:20s} R2={r['R2']:.3f}  MAE=Rs.{r['MAE']:,.0f}  RMSE=Rs.{r['RMSE']:,.0f}  CV_R2={r['CV_R2']:.3f}")
print(f"\nBest model: {b['best_name']}  -> saved to {ml.MODEL_PATH}")
