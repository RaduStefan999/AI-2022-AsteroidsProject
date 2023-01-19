import os
from ml_manager import MLManager
from Models.generic_ml_model import GenericMLModel
from Utils.data_container import DataContainer
from Utils.data_loader import DataLoader
from collections.abc import Callable
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


class TrainingPlotMaker:
    def __init__(self, trained_model_name: str):
        self.trained_model_name = trained_model_name

    def initialize(self) -> None:
        if not os.path.isdir(f"MLComparer/{self.trained_model_name}"):
            os.mkdir(f"MLComparer/{self.trained_model_name}")

    def make_plot(self, trained_model_specs: dict[str, list]) -> None:
        self.initialize()
        for specs_name, model_specs in trained_model_specs.items():
            model_specs_frame = {"iteration": list(range(len(model_specs))), f"{specs_name}": model_specs}
            current_bar_plot = sns.barplot(x="iteration", y=f"{specs_name}", data=model_specs_frame)
            current_bar_plot.set(ylabel=f"{specs_name}")

            plt.savefig(f"MLComparer/{self.trained_model_name}/{specs_name}.png")
            plt.clf()


class BenchmarkPlotMaker:
    def __init__(self):
        pass

    def make_plot(self, trained_model_specs: dict[str, dict[str, float]]) -> None:
        algorithm_specs_comparison = dict()

        for model_name, specs_dict in trained_model_specs.items():
            for spec_name, spec_value in specs_dict.items():
                algorithm_spec_item = algorithm_specs_comparison.get(spec_name, {"model_name": [], "spec_value": []})
                algorithm_spec_item["model_name"] += [model_name]
                algorithm_spec_item["spec_value"] += [spec_value]

                algorithm_specs_comparison[spec_name] = algorithm_spec_item

        for spec_name, model_values_obj in algorithm_specs_comparison.items():
            model_values_obj_pd_dataframe = pd.DataFrame.from_dict(model_values_obj)

            sorted_model_values_obj_pd_dataframe = model_values_obj_pd_dataframe.sort_values(by="spec_value", ascending=False)

            plt.subplots(figsize=(16, 8))
            fig, ax = plt.subplots()
            fig.subplots_adjust(bottom=0.4)

            current_bar_plot = sns.barplot(x="model_name", y=f"spec_value", data=sorted_model_values_obj_pd_dataframe)

            current_bar_plot.set(ylabel=f"{spec_name}")

            plt.xticks(rotation=60)
            plt.savefig(f"MLComparer/models_comparison_on_{spec_name}.png")
            plt.clf()


class MLComparer:
    def __init__(self):
        self.data_container = DataContainer()

        self.all_models_training_specs = dict()
        self.all_models_benchmark_specs = dict()

    def initialize(self) -> None:
        self.data_container.load(DataLoader("./Data/Engineered/Asteroid_Updated_Engineered_Scaled.bin"))

    def compare_training(self, model_name: str, model: GenericMLModel, number_of_epochs: int,
                         override_train_model_get_specs: Callable = None) -> GenericMLModel:

        if override_train_model_get_specs is not None:
            trained_model, training_specs = override_train_model_get_specs(model, self.data_container,
                                                                           number_of_epochs)

        else:
            trained_model, training_specs = MLManager.train_model_get_specs(model, self.data_container,
                                                                            number_of_epochs)

        self.all_models_training_specs[model_name] = training_specs
        return trained_model

    def compare_benchmark(self, model_name: str, model: GenericMLModel) -> None:
        benchmark_specs = MLManager.benchmark_model(model, self.data_container)

        self.all_models_benchmark_specs[model_name] = benchmark_specs

    def dump_comparison(self) -> None:
        for trained_model_name, trained_model_specs in self.all_models_training_specs.items():
            training_plot_maker_obj = TrainingPlotMaker(trained_model_name)
            training_plot_maker_obj.make_plot(trained_model_specs)

        benchmark_plot_maker_obj = BenchmarkPlotMaker()
        benchmark_plot_maker_obj.make_plot(self.all_models_benchmark_specs)