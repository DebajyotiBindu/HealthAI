
from src.model_manager import MODEL_FILES, load_model

for disease in MODEL_FILES:
    print("\n" + "=" * 65)
    print(f"DISEASE: {disease}")
    print("=" * 65)

    model = load_model(disease)
    classifier = model.named_steps["model"]

    print("\nClassifier:", type(classifier).__name__)
    print("Model classes:", classifier.classes_)

    print("\nPipeline steps:")
    for name, step in model.named_steps.items():
        print(f"  {name}: {type(step).__name__}")

    print("\nCategorical encoder details:")

    preprocessor = model.named_steps["preprocessor"]

    transformers = getattr(preprocessor, "transformers_", [])

    found_encoder = False

    for name, transformer, columns in transformers:
        if name == "remainder" or transformer == "drop":
            continue

        encoder = transformer

        if hasattr(encoder, "named_steps"):
            for step in encoder.named_steps.values():
                if hasattr(step, "categories_"):
                    encoder = step
                    break

        if hasattr(encoder, "categories_"):
            found_encoder = True

            print(f"\n  Transformer: {name}")
            print("  Columns:", list(columns))

            for column, categories in zip(
                columns, encoder.categories_
            ):
                print(f"    {column}: {list(categories)}")

    if not found_encoder:
        print("  No directly accessible categorical encoder found.")
