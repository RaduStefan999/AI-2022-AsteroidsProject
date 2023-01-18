from Models.RNModels.rn_model import RNModel
from Models.ADABoostModels.ada_boost_model import ADABoostModel
from Models.RandomForestModels.random_forest_model import RandomForestModel
from Models.SVMModel.svm_model import SVMModel
from Models.KNNModels.knn_model import KNNModel
from ml_comparer import MLComparer
from knn_ml_manager import KNNManager


if __name__ == '__main__':
    ml_comparer_obj = MLComparer()
    ml_comparer_obj.initialize()

    # trained_rn_model = ml_comparer_obj.compare_training("RM_Model", RNModel(), 10)
    # ml_comparer_obj.compare_benchmark("RM_Model", trained_rn_model)
    #
    # trained_adaboost_model = ml_comparer_obj.compare_training("ADABoost_Model", ADABoostModel(), 10)
    # ml_comparer_obj.compare_benchmark("ADABoost_Model", trained_adaboost_model)
    #
    # trained_svm_model = ml_comparer_obj.compare_training("SVM_Model", SVMModel(), 10)
    # ml_comparer_obj.compare_benchmark("SVM_Model", trained_svm_model)
    #
    # trained_random_forest_model = ml_comparer_obj.compare_training("RandomForest_Model", RandomForestModel(), 10)
    # ml_comparer_obj.compare_benchmark("RandomForest_Model", trained_random_forest_model)

    # Compare training of KNN models with different k values

    trained_knn_model = ml_comparer_obj.compare_training("KNN_Model", KNNModel(),  15, KNNManager.train_model_get_specs)

    ml_comparer_obj.compare_benchmark("KNN_Model", trained_knn_model)

    ml_comparer_obj.dump_comparison()


