from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor

class BaseModel():
    def __new__(*args, **kwargs):
        seed = kwargs['seed'] if 'seed' in kwargs else args[0] if len(args) > 0 else 0

        hidden_layer_sizes = (100, 20)
        n_models = 5

        models = []
        for i in range(n_models):
            models.append(MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, solver = "sgd", activation = "relu", random_state=seed+i))

        # Combine them in a VotingRegressor
        return VotingRegressor(estimators=[(f'model_{i}', model) for i, model in enumerate(models)])

if __name__ == '__main__':
    model = BaseModel(seed=0)
    print(model.estimators[0][1])
    print(f"Estimator type : {model._estimator_type}")
    print(f"Number of estimators : {len(model.estimators)}")