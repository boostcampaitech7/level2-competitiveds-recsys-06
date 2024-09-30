import pandas as pd

from src.model.simplemodel import SimpleModel


def app():
    model = SimpleModel()
    print("Model Train Start.")
    model.train()
    _, test = model.get_train_test_data()
    print("Model Predict Start.")
    test["target"] = model.predict()
    test.to_csv("output.csv", index=False)


if __name__ == "__main__":
    app()
