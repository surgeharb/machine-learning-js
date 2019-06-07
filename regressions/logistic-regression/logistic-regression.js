const _ = require('lodash');
const tf = require('@tensorflow/tfjs');

const DEFAULT_OPTIONS = {
  decisionBoundary: 0.5,
  learningRate: 0.1,
  iterations: 1000,
  batchSize: 10,
};

class LogisticRegression {

  constructor(features, labels, options = {}) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.costHistory = [];

    this.options = { ...DEFAULT_OPTIONS, ...options };

    this.weights = tf.zeros([this.features.shape[1], 1]);
  }

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights).sigmoid();
    const differences = currentGuesses.sub(labels);

    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0])

    this.weights =
      this.weights.sub(slopes.mul(this.options.learningRate));
  }

  processFeatures(features) {
    features = tf.tensor(features);
    features = this.standardize(features);

    return tf.ones([features.shape[0], 1]).concat(features, 1);
  }

  standardize(features) {
    if (!this.mean || !this.variance) {
      const { mean, variance } = tf.moments(features, 0);

      this.variance = variance.add(1e-7);
      this.mean = mean;
    }

    return features.sub(this.mean).div(this.variance.pow(0.5));
  }

  // Record Cross Entropy History
  recordCost() {
    const guesses = this.features.matMul(this.weights).sigmoid();

    const termOne = this.labels.transpose().matMul(guesses.log());
    
    const termTwo = this.labels
      .mul(-1).add(1)
      .transpose()
      .matMul(
        guesses.mul(-1).add(1).log()
      );

    const cost = termOne.add(termTwo)
      .div(this.features.shape[0])
      .mul(-1)
      .bufferSync()
      .get(0, 0);

    this.costHistory.unshift(cost);
  }

  updateLearningRate() {
    if (this.costHistory.length < 2) {
      return;
    }

    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }

  train() {
    const { batchSize } = this.options;

    const batchQuantity = Math.floor(
      this.features.shape[0] / batchSize
    );

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * batchSize;
        const featureSlice = this.features.slice(
          [startIndex, 0],
          [batchSize, -1]
        );

        const labelSlice = this.labels.slice(
          [startIndex, 0],
          [batchSize, -1]
        );

        this.gradientDescent(featureSlice, labelSlice);
      }

      this.recordCost();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    return this.processFeatures(observations)
      .matMul(this.weights)
      .sigmoid()
      .greaterEqual(this.options.decisionBoundary)
      .cast('float32'); // old tensorflow greater return boolean instead of 0 & 1
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels);

    const incorrect = predictions
      .sub(testLabels)
      .abs()
      .sum()
      .bufferSync()
      .get();

    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

}

module.exports = LogisticRegression;