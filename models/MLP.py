import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor

class MLP():
    def __new__(*args, **kwargs):
        seed = kwargs['seed'] if 'seed' in kwargs else args[0] if len(args) > 0 else 0
        max_iter = kwargs['max_iter'] if 'max_iter' in kwargs else 1000
        early_stopping = kwargs['early_stopping'] if 'early_stopping' in kwargs else True
        n_models = kwargs['n_models'] if 'n_models' in kwargs else 5

        hidden_layer_sizes = (100, 20)

        models = []
        for i in range(n_models):
            models.append(MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes, 
                solver="adam", 
                activation="relu", 
                early_stopping=early_stopping, 
                max_iter=max_iter, 
                random_state=seed+i
        ))

        # Combine them in a VotingRegressor
        return VotingRegressor(estimators=[(f'model_{i}', model) for i, model in enumerate(models)])

if __name__ == '__main__':
    model = MLP(seed=0)
    print(model.estimators[0][1])
    print(f"Estimator type : {model._estimator_type}")
    print(f"Number of estimators : {len(model.estimators)}")