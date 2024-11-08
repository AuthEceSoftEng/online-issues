from river import feature_extraction, compose, naive_bayes, multiclass, ensemble, drift, metrics
from collections import defaultdict

class OnlineModel():
    def __init__(self, drift_detector_delta = None, variables = "all"):
        if variables == "all":
            variables = ("summary", "description", "labels", "components_name", "priority_name", "issue_type_name")
        if drift_detector_delta is not None:
            self.drift_detector = drift.ADWIN(delta = drift_detector_delta)
        else:
            self.drift_detector = None

        # Perform TFIDF on the text features of the dataset
        if type(variables) == str: variables = [variables]
        self.transformer = compose.TransformerUnion(
            *[(v, feature_extraction.TFIDF(on=v)) for v in variables]
        )

        self.drifts = []
        self.accuracies = []
        self.accuracy = metrics.Accuracy()

    def _transform(self, x):
        # Apply the transformer
        self.transformer.learn_one(x)
        x = self.transformer.transform_one(x)
        return x

    def _apply_model(self, x, y):
        # Apply the model (predict the next instance and then train the model on it)
        y_pred = self.my_model.predict_one(x)
        self.my_model.learn_one(x, y)
        return y_pred

    def _calculate_accuracy(self, y, y_pred):
        # Update the accuracy
        self.accuracy.update(y, y_pred)
        self.accuracies.append(self.accuracy.get())

    def _update_drifts(self, i):
        # Check if drift is detected
        if self.drift_detector is not None:
            self.drift_detector.update(self.accuracy.get())
            if self.drift_detector.drift_detected:
                self.drifts.append(i)

    def apply_model(self, i, x, y):
        x = self._transform(x)
        y_pred = self._apply_model(x, y)
        self._calculate_accuracy(y, y_pred)
        self._update_drifts(i)
        return y_pred

    def results(self):
        return {"accuracies": self.accuracies, "drifts": self.drifts}

class NaiveBayesOnlineModel(OnlineModel):
    def __init__(self, alpha=0.1, variables="all"):
        super(NaiveBayesOnlineModel, self).__init__(variables=variables)
        self.name = "Naive Bayes"
        if variables != "all": self.name += "_" + ('_'.join(variables) if type(variables) != str else variables)
        self.my_model = naive_bayes.MultinomialNB(alpha=alpha)

class NaiveBayesWithADWINOnlineModel(OnlineModel):
    def __init__(self, alpha=0.1, delta=0.15, variables="all"):
        super(NaiveBayesWithADWINOnlineModel, self).__init__(drift_detector_delta=delta, variables=variables)
        self.name = "Naive Bayes with ADWIN"
        if variables != "all": self.name += "_" + ('_'.join(variables) if type(variables) != str else variables)
        self.my_model = naive_bayes.MultinomialNB(alpha=alpha)

    def apply_model(self, i, x, y):
        x = self._transform(x)
        y_pred = self._apply_model(x, y)
        self._calculate_accuracy(y, y_pred)
        if self.drift_detector is not None:
            self.drift_detector.update(self.accuracy.get())
            if self.drift_detector.drift_detected:
                self.drifts.append(i)
                self.my_model = self.my_model.clone()
        return y_pred

class AdaboostOnlineModel(OnlineModel):
    def __init__(self, alpha=0.1, n_models=10, variables="all"):
        super(AdaboostOnlineModel, self).__init__(variables=variables)
        self.name = "Adaboost"
        if variables != "all": self.name += "_" + ('_'.join(variables) if type(variables) != str else variables)
        base_model = multiclass.OneVsRestClassifier(naive_bayes.MultinomialNB(alpha = alpha))
        self.my_model = ensemble.AdaBoostClassifier(
            model = base_model,
            n_models = n_models, # number of base classifiers
            seed = 42
        )

class AdaboostWithADWINOnlineModel(OnlineModel):
    def __init__(self, alpha=0.1, delta=0.15, n_models=10, variables="all"):
        super(AdaboostWithADWINOnlineModel, self).__init__(drift_detector_delta=delta, variables=variables)
        self.name = "Adaboost with ADWIN"
        if variables != "all": self.name += "_" + ('_'.join(variables) if type(variables) != str else variables)
        self.n_models = n_models
        base_model = multiclass.OneVsRestClassifier(naive_bayes.MultinomialNB(alpha = alpha))
        self.my_model = ensemble.AdaBoostClassifier(
            model = base_model,
            n_models = n_models, # number of base classifiers
            seed = 42
        )
        self.basemodelaccuracies = [metrics.Accuracy()] * n_models
        self.devweights = defaultdict(lambda: 1)
        self.last100 = []

    def _apply_model(self, x, y):
        self.last100.append(y)
        if len(self.last100) >= 100:
            self.last100 = self.last100[-100:]
            self.devweights = defaultdict(lambda: 0)
            for dev in set(self.last100):
                self.devweights[dev] = 1
        
        # Apply the model (predict the next instance and then train the model on it)
        y_pred_proba = self.my_model.predict_proba_one(x)
        if y_pred_proba:
            #y_pred = max(y_pred_proba, key=y_pred_proba.get)
            y_pred = sorted(((res_key, res_value * self.devweights[res_key]) for res_key, res_value in y_pred_proba.items()), key = lambda xx: -xx[1])[0][0]
        else:
            y_pred = None

        self.my_model.learn_one(x, y)
        return y_pred

    def apply_model(self, i, x, y):
        x = self._transform(x)
        y_pred = self._apply_model(x, y)
        self._calculate_accuracy(y, y_pred)
        for index_of_model, acc in enumerate(self.basemodelaccuracies):            # Update
            permodel_y_pred = self.my_model.models[index_of_model].predict_one(x)  # accuracies
            acc.update(y, permodel_y_pred)                                         # of base models
        if self.drift_detector is not None:
            self.drift_detector.update(self.accuracy.get())
            if self.drift_detector.drift_detected:
                self.drifts.append(i)
                models_and_accuracies_sorted = list(sorted([(i, acc.get()) for i, acc in enumerate(self.basemodelaccuracies)], key = lambda x: x[1]))
                for index_of_model, _ in models_and_accuracies_sorted[:self.n_models // 2]:
                    self.my_model.models[index_of_model] = self.my_model.model.clone()
                    self.my_model.correct_weight[index_of_model] = 0
                    self.my_model.wrong_weight[index_of_model] = 0
#                    for everyx, everyy in last100data:
#                        self.my_model.models[index_of_model].learn_one(everyx, everyy)
                    
        return y_pred

class EnhancedOnlineModel(OnlineModel):
    def __init__(self, alpha=0.1, delta=0.15, n_models=10, variables="all"):
        super(EnhancedOnlineModel, self).__init__(drift_detector_delta=delta, variables=variables)
        self.name = "Enhanced Online Model"
        if variables != "all": self.name += "_" + ('_'.join(variables) if type(variables) != str else variables)
        self.n_models = n_models
        base_model = multiclass.OneVsRestClassifier(naive_bayes.MultinomialNB(alpha = alpha))
        self.my_model = ensemble.AdaBoostClassifier(
            model = base_model,
            n_models = n_models, # number of base classifiers
            seed = 42
        )
        self.basemodelaccuracies = [metrics.Accuracy()] * n_models
        self.devweights = defaultdict(lambda: 1)
        self.last100 = []

    def _apply_model(self, x, y):
        self.last100.append(y)
        if len(self.last100) >= 100:
            self.last100 = self.last100[-100:]
            self.devweights = defaultdict(lambda: 0)
            for dev in set(self.last100):
                self.devweights[dev] = 1
        
        # Apply the model (predict the next instance and then train the model on it)
        y_pred_proba = self.my_model.predict_proba_one(x)
        if y_pred_proba:
            #y_pred = max(y_pred_proba, key=y_pred_proba.get)
            y_pred = sorted(((res_key, res_value * self.devweights[res_key]) for res_key, res_value in y_pred_proba.items()), key = lambda xx: -xx[1])[0][0]
        else:
            y_pred = None

        self.my_model.learn_one(x, y)
        return y_pred

    def apply_model(self, i, x, y):
        x = self._transform(x)
        y_pred = self._apply_model(x, y)
        self._calculate_accuracy(y, y_pred)
        for index_of_model, acc in enumerate(self.basemodelaccuracies):            # Update
            permodel_y_pred = self.my_model.models[index_of_model].predict_one(x)  # accuracies
            acc.update(y, permodel_y_pred)                                         # of base models
        if self.drift_detector is not None:
            self.drift_detector.update(self.accuracy.get())
            if self.drift_detector.drift_detected:
                self.drifts.append(i)
                models_and_accuracies_sorted = list(sorted([(i, acc.get()) for i, acc in enumerate(self.basemodelaccuracies)], key = lambda x: x[1]))
                for index_of_model, _ in models_and_accuracies_sorted[:self.n_models // 2]:
                    self.my_model.models[index_of_model] = self.my_model.model.clone()
                    self.my_model.correct_weight[index_of_model] = 0
                    self.my_model.wrong_weight[index_of_model] = 0
#                    for everyx, everyy in last100data:
#                        self.my_model.models[index_of_model].learn_one(everyx, everyy)
                    
        return y_pred
