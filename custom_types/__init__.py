from typing import TypedDict
import numpy as np

# TestDataType = Dict['keyword':np.array, 'location':np.array, 'text':np.array]
# TestDataType = Dict['keyword':np.array, 'location':np.array, 'text':np.array]


class TrainDataType(TypedDict):
    keyword: np.array
    location: np.array
    text: np.array
    hashtags: np.array
    target: np.array
    id: np.array


class TestDataType(TypedDict):
    keyword: np.array
    location: np.array
    text: np.array
    hashtags: np.array
    id: np.array