import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor

sklearn.set_config(enable_metadata_routing=True)

class BaseModel():
    def __new__(*args, **kwargs):
        seed = kwargs['seed'] if 'seed' in kwargs else args[0] if len(args) > 0 else 0
        max_iter = kwargs['max_iter'] if 'max_iter' in kwargs else 1000
        early_stopping = kwargs['early_stopping'] if 'early_stopping' in kwargs else True

        hidden_layer_sizes = (100, 20)
        n_models = 5

        models = []
        for i in range(n_models):
            models.append(MLPRegressor(
                learning_rate_init=1e-3,
                alpha=1e-5,
                hidden_layer_sizes=hidden_layer_sizes, 
                solver="adam", 
                activation="relu", 
                early_stopping=early_stopping, 
                max_iter=max_iter, 
                random_state=seed+i))

        # Combine them in a VotingRegressor
        return VotingRegressor(estimators=[(f'model_{i}', model) for i, model in enumerate(models)])

if __name__ == '__main__':
    model = BaseModel(seed=0)
    print(model.estimators[0][1])
    print(f"Estimator type : {model._estimator_type}")
    print(f"Number of estimators : {len(model.estimators)}")