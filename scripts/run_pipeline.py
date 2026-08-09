from mushroom_optimizer.data_collection import generate_demo_data
from mushroom_optimizer.config import RAW_DATA_PATH
from mushroom_optimizer.train import run_training


def main() -> None:
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_demo_data()
    frame.to_csv(RAW_DATA_PATH, index=False)
    print(f"1/2 Collected {len(frame):,} grow-room observations")
    metrics = run_training(RAW_DATA_PATH)
    print("2/2 Trained and evaluated both models")
    print(f"Yield MAE: {metrics['yield_model']['mae_kg']:.2f} kg")
    print(f"Contamination ROC AUC: {metrics['contamination_model']['roc_auc']:.3f}")


if __name__ == "__main__":
    main()

