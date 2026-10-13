
from pathlib import Path
import pickle

MODEL_DIR = Path("MODELS")


def inspect_model(path):
    print("\n" + "=" * 60)
    print(f"FILE: {path.name}")
    print("=" * 60)

    try:
        with path.open("rb") as file:
            model = pickle.load(file)

        print("Model type:", type(model).__name__)

        if hasattr(model, "n_features_in_"):
            print("Number of features:", model.n_features_in_)

        if hasattr(model, "feature_names_in_"):
            print("Feature names:")
            for i, name in enumerate(model.feature_names_in_, 1):
                print(f"  {i}. {name}")

        if hasattr(model, "classes_"):
            print("Classes:", model.classes_)

        if hasattr(model, "named_steps"):
            print("Pipeline steps:")
            for name, step in model.named_steps.items():
                print(f"  {name}: {type(step).__name__}")

    except Exception as error:
        print(f"Inspection failed: {type(error).__name__}: {error}")


def main():
    files = sorted(MODEL_DIR.glob("*.pkl"))

    if not files:
        print("No .pkl files found. Check the MODELS folder.")
        return

    print(f"Found {len(files)} model files.")

    for path in files:
        inspect_model(path)


if __name__ == "__main__":
    main()
