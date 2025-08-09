def evaluate(model, criteria, X, y):
    return criteria(model.predict(X), y)