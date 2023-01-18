from Models.RNModels.rn_model import RNModel
from Models.ADABoostModels.ada_boost_model import ADABoostModel
from Models.RandomForestModels.random_forest_model import RandomForestModel
from Models.SVMModel.svm_model import SVMModel
from ml_comparer import MLComparer


if __name__ == '__main__':
    ml_comparer_obj = MLComparer()
    ml_comparer_obj.initialize()

    trained_rn_model = ml_comparer_obj.compare_training("RN_Model", RNModel(), 10)
    ml_comparer_obj.compare_benchmark("RN_Model", trained_rn_model)

    # trained_adaboost_model = ml_comparer_obj.compare_training("ADABoost_Model", ADABoostModel(), 10)
    # ml_comparer_obj.compare_benchmark("ADABoost_Model", trained_adaboost_model)

    # trained_svm_model = ml_comparer_obj.compare_training("SVM_Model", SVMModel(), 10)
    # ml_comparer_obj.compare_benchmark("SVM_Model", trained_svm_model)

    # trained_random_forest_model = ml_comparer_obj.compare_training("RandomForest_Model", RandomForestModel(), 10)
    # ml_comparer_obj.compare_benchmark("RandomForest_Model", trained_random_forest_model)
    
    ml_comparer_obj.dump_comparison()


