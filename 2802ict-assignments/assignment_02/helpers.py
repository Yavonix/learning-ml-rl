from dataclasses import dataclass, field
from functools import reduce

@dataclass
class Metric:
    TP: int = 0
    FP: int = 0
    TN: int = 0
    FN: int = 0

    def total(self) -> int: ## equal to the total number of samples in the test set
        return self.TP + self.FP + self.TN + self.FN

    def accuracy(self) -> float:
        if self.total() == 0: return 0.0
        return (self.TP + self.TN) / self.total()
    
    def precision(self ) -> float:
        denominator = self.TP + self.FP
        if denominator == 0: return 0.0
        return self.TP / denominator
    
    def recall(self) -> float:
        denominator = self.TP + self.FN
        if denominator == 0: return 0.0
        return self.TP / denominator
    
    def f1_score(self) -> float:
        prec = self.precision()
        recall = self.recall()
        denominator = prec + recall
        if denominator == 0: return 0.0
        return 2 * prec * recall / denominator
    
    def support(self) -> int: # the count of this class (ie unacc)
        return self.TP + self.FN

class Metrics():
    def __init__(self, classes: list[str]):
        self.metrics = {i:Metric() for i in classes}

    def update(self, model_prediction: str, ground_truth: str):
        for cls in self.metrics:
            if model_prediction == cls and cls == ground_truth: # model predicted class and was class was right
                self.metrics[cls].TP += 1
            elif model_prediction != cls and cls != ground_truth: # model did not predict class and class was not right
                self.metrics[cls].TN += 1
            elif model_prediction == cls and model_prediction != ground_truth: # model predicted class and class was not right
                self.metrics[cls].FP += 1
            elif model_prediction != cls and cls == ground_truth: # model did not predict class and class was right
                self.metrics[cls].FN += 1

    def print_summary(self):
        print(f"{'':<20} {'precision':<15} {'recall':<15} {'f1-score':<15} {'support':<15}")
        print("-" * 80)

        for key, value in self.metrics.items():
            print(f"{key:<20} {value.precision():<15.4f} {value.recall():<15.4f} {value.f1_score():<15.4f} {value.support():<15}")

        print("-" * 80)
        
        avg_results = self.average_results()
        total = self.total()

        print(f"{'macro avg':<20} {avg_results['macro_precision']:<15.4f} {avg_results['macro_recall']:<15.4f} {avg_results['macro_f1_score']:<15.4f} {total:<15}")
        print(f"{'weighted avg':<20} {avg_results['weighted_precision']:<15.4f} {avg_results['weighted_recall']:<15.4f} {avg_results['weighted_f1_score']:<15.4f} {total:<15}")
        
        print("-" * 80)

        print()

        print(f"{"accuracy":<20} {self.accuracy():.4f}")

    def total(self) -> int:
        return next(iter(self.metrics.values())).total()
        # this also works but would be slower:
        # return sum(map(lambda k: k.support(), self.metrics.values()))

    def accuracy(self) -> float:
        return sum(map(lambda k: k.TP, self.metrics.values())) / self.total()
    
    def average_results(self) -> dict[str, float]:
        scores: dict[str, float] = {}

        total = self.total()

        scores["macro_precision"] = sum([k.precision() for k in self.metrics.values()])/len(self.metrics)
        scores["weighted_precision"] = sum([k.precision()*k.support() for k in self.metrics.values()])/total

        scores["macro_recall"] = sum([k.recall() for k in self.metrics.values()])/len(self.metrics)
        scores["weighted_recall"] = sum([k.recall()*k.support() for k in self.metrics.values()])/total

        scores["macro_f1_score"] = sum([k.f1_score() for k in self.metrics.values()])/len(self.metrics)
        scores["weighted_f1_score"] = sum([k.f1_score()*k.support() for k in self.metrics.values()])/total

        return scores