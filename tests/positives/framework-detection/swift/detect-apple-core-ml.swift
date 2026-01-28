// Positive test cases for Apple Core ML (Swift)

// ruleid: detect-apple-core-ml
import CoreML

// ruleid: detect-apple-core-ml
import Vision

// ruleid: detect-apple-core-ml
let config = MLModelConfiguration()

// ruleid: detect-apple-core-ml
let model = try MLModel.load(contentsOf: modelURL)

// ruleid: detect-apple-core-ml
let vnModel = try VNCoreMLModel(for: model)

// ruleid: detect-apple-core-ml
let prediction = try model.prediction(from: input)
