
from pathlib import Path
import pickle

MODEL_DIR = Path("MODELS")


def inspect_pipeline(path):
    print("\n" + "=" * 75)
    print(f"MODEL: {path.name}")
    print("=" * 75)

    with path.open("rb") as file:
        pipeline = pickle.load(file)

    print("Pipeline steps:", list(pipeline.named_steps.keys()))

    preprocessor = pipeline.named_steps.get("preprocessor")

    if preprocessor is None:
        print("No preprocessor step found.")
        return

    print("\nColumn transformers:")

    for name, transformer, columns in preprocessor.transformers_:
        print(f"\nTransformer: {name}")
        print("Type:", type(transformer).__name__)
        print("Columns:", columns)

        if hasattr(transformer, "categories_"):
            print("Learned categories:")
            for column, categories in zip(
                columns, transformer.categories_
            ):
                print(f"  {column}: {categories}")

        if hasattr(transformer, "statistics_"):
            print("Imputer statistics:", transformer.statistics_)

    model = pipeline.named_steps.get("model")

    if model is not None:
        print("\nFinal estimator:", type(model).__name__)

        if hasattr(model, "get_params"):
            params = model.get_params(deep=False)
            print("Estimator parameters:")
            for key, value in params.items():
                if key in {
                    "random_state",
                    "max_depth",
                    "n_estimators",
                    "activation",
                    "solver",
                    "C",
                }:
                    print(f"  {key}: {value}")


def main():
    for path in sorted(MODEL_DIR.glob("*.pkl")):
        try:
            inspect_pipeline(path)
        except Exception as error:
            print(f"\nFailed: {path.name}: {error}")


if __name__ == "__main__":
    main()
