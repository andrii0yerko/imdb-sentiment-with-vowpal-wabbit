import joblib


class SemanticModel:
    '''
    Wrapper around the estimator pipeline,
    which provides functions to get predictions and input weights
    in convenient format.
    '''

    def __init__(self, pipeline_path):
        model_dict = joblib.load(pipeline_path)
        self.pipeline = model_dict['pipeline']
        self.version = model_dict['tag']
        self.comment = model_dict['comment']
        self.class_names = ['negative', 'positive']

    def predict(self, data):
        pred = self.pipeline.predict_proba(data)
        pred_dict = [dict(zip(self.class_names, x)) for x in pred]
        if len(pred_dict) == 1:
            pred_dict = pred_dict[0]
        return pred_dict

    def weights(self, data):
        preprocessor = self.pipeline['preprocessor']
        vw_model = self.pipeline['classifier'].vw_
        feature_space = self.pipeline['namespace_adder'].feature_space
        original, preprocessed = preprocessor.mark_up_text(data, 2)

        # WARNING
        # TODO it should be replaced with the native get_weight_from_name implementation
        # as soon as the new python package version was released (or install it from the repo directly...)
        # weights = [vw_model.get_weight_from_name(x, feature_space) for x in preprocessed]
        def get_weight_from_name(model, name, namespace=""):
            return model.get_weight(model.hash_feature(name, model.hash_space(namespace)))

        weights = [get_weight_from_name(vw_model, x, feature_space) for x in preprocessed]
        return dict(zip(original, weights))

    @property
    def info(self):
        return {'version': self.version, 'info': self.comment}

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"version={self.version!r}, "
            f"{self.pipeline!r}, "
            f"info={self.comment!r}"
        )
