from pydantic import BaseModel, Field
from typing import List, Dict
import numpy as np

class ARTModule(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    input_size: int
    category_size: int
    vigilance: float = Field(0.9, ge=0.0, le=1.0)
    weights: np.ndarray = Field(default_factory=lambda: np.array([]))

    def __init__(self, **data):
        super().__init__(**data)
        self.weights = np.ones((self.category_size, self.input_size))

    def activate(self, input_vector: np.ndarray) -> int:
        match_scores = np.minimum(input_vector, self.weights).sum(axis=1) / (self.weights.sum(axis=1) + 1e-10)
        best_match = np.argmax(match_scores)
        if match_scores[best_match] >= self.vigilance:
            return best_match
        return -1  # No match found

    def learn(self, input_vector: np.ndarray, category: int):
        self.weights[category] = np.minimum(input_vector, self.weights[category])

class MapField(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    arta_size: int
    artb_size: int
    weights: np.ndarray = Field(default_factory=lambda: np.array([]))

    def __init__(self, **data):
        super().__init__(**data)
        self.weights = np.zeros((self.arta_size, self.artb_size))

    def learn(self, arta_category: int, artb_category: int):
        self.weights[arta_category, artb_category] = 1

    def predict(self, arta_category: int) -> int:
        return np.argmax(self.weights[arta_category])

class ARTMAP(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    arta: ARTModule
    artb: ARTModule
    map_field: MapField
    learning_rate: float = Field(0.5, ge=0.0, le=1.0)

    def train(self, input_vector: np.ndarray, target_vector: np.ndarray):
        arta_category = self.arta.activate(input_vector)
        artb_category = self.artb.activate(target_vector)

        if arta_category == -1:
            arta_category = np.argmin(self.arta.weights.sum(axis=1))
        if artb_category == -1:
            artb_category = np.argmin(self.artb.weights.sum(axis=1))

        predicted_artb = self.map_field.predict(arta_category)

        if predicted_artb != artb_category:
            self.arta.vigilance = min(self.arta.vigilance + 0.1, 0.9)  # Increase vigilance, but cap it
            return self.train(input_vector, target_vector)  # Recursive call

        self.arta.learn(input_vector, arta_category)
        self.artb.learn(target_vector, artb_category)
        self.map_field.learn(arta_category, artb_category)
        
        print(f"Trained: Input {input_vector} -> ARTa category {arta_category} -> ARTb category {artb_category}")

    def predict(self, input_vector: np.ndarray) -> int:
        arta_category = self.arta.activate(input_vector)
        if arta_category == -1:
            return -1  # Unable to classify
        predicted_artb = self.map_field.predict(arta_category)
        print(f"Prediction: Input {input_vector} -> ARTa category {arta_category} -> ARTb category {predicted_artb}")
        return predicted_artb

    def predict_class(self, input_vector: np.ndarray) -> np.ndarray:
        artb_category = self.predict(input_vector)
        if artb_category == -1:
            return np.zeros(self.artb.input_size)  # Unable to classify
        return np.eye(self.artb.input_size)[artb_category]

# Example usage
artmap = ARTMAP(
    arta=ARTModule(input_size=4, category_size=10, vigilance=0.8),
    artb=ARTModule(input_size=3, category_size=3, vigilance=0.9),
    map_field=MapField(arta_size=10, artb_size=3),
    learning_rate=0.5
)

# Training 1D
input_data = np.array([0.1, 0.2, 0.3, 0.4])
target_data = np.array([1, 0, 0])
artmap.train(input_data, target_data)

# Prediction
new_input = np.array([0.15, 0.25, 0.35, 0.45])

predicted_category = artmap.predict(new_input)
print(f"Predicted category: {predicted_category}")

predicted_class = artmap.predict_class(new_input)
print(f"Predicted class: {predicted_class}")

# Training 2D
print('2D example\n\n')
# Training data
input_data = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.2, 0.3, 0.4, 0.5],
    [0.7, 0.8, 0.9, 1.0]
])

target_data = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
])

# Training loop
for i in range(len(input_data)):
    artmap.train(input_data[i], target_data[i])
    print(f"Trained on example {i+1}")

# Prediction function
def predict_class(artmap, input_vector):
    artb_category = artmap.predict(input_vector)
    if artb_category == -1:
        return np.zeros(artmap.artb.input_size)  # Unable to classify
    return np.eye(artmap.artb.input_size)[artb_category]

# Test data
test_data = np.array([
    [0.15, 0.25, 0.35, 0.45],
    [0.55, 0.65, 0.75, 0.85],
    [0.3, 0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9, 1.0]
])

# Predictions
for i, test_input in enumerate(test_data):
    predicted_class = predict_class(artmap, test_input)
    print(f"Test example {i+1}:")
    print(f"  Input: {test_input}")
    print(f"  Predicted class: {predicted_class}")
    print(f"  Predicted category: {np.argmax(predicted_class)}")
    print()

