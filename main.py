from ml_store_model import MLStoreModel
from Models.RNModels.rn_model import RNModel
from Models.CNNRNModel.cnn_rn_model import CNNRNModel
from Models.ADABoostModels.ada_boost_model import ADABoostModel
from Models.RandomForestModels.random_forest_model import RandomForestModel
from Models.SVMModel.svm_model import SVMModel
from Models.KNNModels.knn_model import KNNModel
from ml_comparer import MLComparer
from knn_ml_manager import KNNManager


def train_and_save() -> None:

    ml_comparer_obj = MLComparer()
    ml_comparer_obj.initialize()

    trained_rn_model = ml_comparer_obj.compare_training("RN_Model", RNModel(), 5)
    ml_comparer_obj.compare_benchmark("RN_Model", trained_rn_model)
    MLStoreModel.store("RN_Model", trained_rn_model)

    trained_cnn_rn_model = ml_comparer_obj.compare_training("CNN_RN_Model", CNNRNModel(), 10)
    ml_comparer_obj.compare_benchmark("CNN_RN_Model", trained_cnn_rn_model)
    MLStoreModel.store("CNN_RN_Model", trained_cnn_rn_model)

    trained_adaboost_model = ml_comparer_obj.compare_training("ADABoost_Model", ADABoostModel(), 10)
    ml_comparer_obj.compare_benchmark("ADABoost_Model", trained_adaboost_model)
    MLStoreModel.store("ADABoost_Model", trained_adaboost_model)

    trained_svm_model = ml_comparer_obj.compare_training("SVM_Model", SVMModel(), 3)
    ml_comparer_obj.compare_benchmark("SVM_Model", trained_svm_model)
    MLStoreModel.store("SVM_Model", trained_svm_model)

    trained_random_forest_model = ml_comparer_obj.compare_training("RandomForest_Model", RandomForestModel(), 10)
    ml_comparer_obj.compare_benchmark("RandomForest_Model", trained_random_forest_model)
    MLStoreModel.store("RandomForest_Model", trained_random_forest_model)

    # Compare training of KNN models with different k values

    trained_knn_uniform_model = ml_comparer_obj.compare_training("KNN_Uniform_Model", KNNModel(weights="uniform"),  15,
                                                                 KNNManager.train_model_get_specs)

    ml_comparer_obj.compare_benchmark("KNN_Uniform_Model", trained_knn_uniform_model)
    MLStoreModel.store("KNN_Uniform_Model", trained_knn_uniform_model)

    trained_knn_distance_model = ml_comparer_obj.compare_training("KNN_Distance_Model", KNNModel(weights="distance"),  15,
                                                                  KNNManager.train_model_get_specs)

    ml_comparer_obj.compare_benchmark("KNN_Distance_Model", trained_knn_distance_model)
    MLStoreModel.store("KNN_Distance_Model", trained_knn_distance_model)

    ml_comparer_obj.dump_comparison()


def load_and_benchmark() -> None:

    ml_comparer_obj = MLComparer()
    ml_comparer_obj.initialize()

    loaded_rn_model = MLStoreModel.load("RN_Model", RNModel())
    ml_comparer_obj.compare_benchmark("RN_Model", loaded_rn_model)

    trained_cnn_rn_model = MLStoreModel.load("CNN_RN_Model", CNNRNModel())
    ml_comparer_obj.compare_benchmark("CNN_RN_Model", trained_cnn_rn_model)

    trained_adaboost_model = MLStoreModel.load("ADABoost_Model", ADABoostModel())
    ml_comparer_obj.compare_benchmark("ADABoost_Model", trained_adaboost_model)

    trained_svm_model = MLStoreModel.load("SVM_Model", SVMModel())
    ml_comparer_obj.compare_benchmark("SVM_Model", trained_svm_model)

    trained_random_forest_model = MLStoreModel.load("RandomForest_Model", RandomForestModel())
    ml_comparer_obj.compare_benchmark("RandomForest_Model", trained_random_forest_model)

    trained_knn_uniform_model = MLStoreModel.load("KNN_Uniform_Model", KNNModel(weights="uniform"))
    ml_comparer_obj.compare_benchmark("KNN_Uniform_Model", trained_knn_uniform_model)

    trained_knn_distance_model = MLStoreModel.load("KNN_Distance_Model", KNNModel(weights="distance"))
    ml_comparer_obj.compare_benchmark("KNN_Distance_Model", trained_knn_distance_model)

    ml_comparer_obj.dump_comparison()


if __name__ == '__main__':

    #train_and_save()
    load_and_benchmark()
