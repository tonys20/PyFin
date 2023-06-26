from abc import ABC, abstractmethod

#Master Strategy Interface
class ModelFramework(ABC):
    @abstractmethod
    def fit(self, data):
        """Fit the model to the data."""
        pass

    @abstractmethod
    def predict(self, new_data):
        """Use the fitted model to make predictions on new data."""
        pass

    @abstractmethod
    def summary(self):
        """Provide a summary of the fitted model."""
        pass

    @abstractmethod
    def cross_validate(self, data, cv):
        """Perform cross validation."""
        pass

    @abstractmethod
    def save_model(self, filename):
        """Save the model to a file."""
        pass

    @abstractmethod
    def load_model(self, filename):
        """Load the model from a file."""
        pass

    @abstractmethod
    def feature_importance(self):
        """Get feature importance for the model."""
        pass

    @abstractmethod
    def get_residuals(self):
        """Get the residuals of the model."""
        pass

    @abstractmethod
    def predict_interval(self, new_data, alpha):
        """Get prediction interval for new data."""
        pass

    @abstractmethod
    def tune_hyperparameters(self, data, param_grid):
        """Tune the hyperparameters of the model."""
        pass

    @abstractmethod
    def plot_diagnostics(self):
        """Plot diagnostic charts for the model."""
        pass

    @abstractmethod
    def check_multicollinearity(self):
        """Check for multicollinearity in the data."""
        pass

    @abstractmethod
    def get_regularization_path(self):
        """Get the regularization path for the model."""
        pass

