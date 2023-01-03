from Models.RNModels.rn_model import RNModel
from Models.ADABoostModels.ada_boost_model import ADABoostModel
from ml_comparer import MLComparer

if __name__ == '__main__':
    ml_comparer_obj = MLComparer()
    ml_comparer_obj.initialize()

    trained_rn_model = ml_comparer_obj.compare_training("rn_model", RNModel(), 15)
    ml_comparer_obj.compare_benchmark("rn_model", trained_rn_model)

    trained_adaboost_model = ml_comparer_obj.compare_training("adaboost_model", ADABoostModel(), 10)
    ml_comparer_obj.compare_benchmark("adaboost_model", trained_adaboost_model)


