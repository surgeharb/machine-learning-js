const _ = require('lodash');
const tf = require('@tensorflow/tfjs');

const DEFAULT_OPTIONS = {
  learningRate: 0.1,
  iterations: 1000,
  batchSize: 10,
};

class LinearRegression {

  constructor(features, labels, options = {}) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.mseHistory = [];

    this.options = { DEFAULT_OPTIONS, ...options };

    this.weights = tf.zeros([this.features.shape[1], 1]);
  }

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights);
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

  recordMSE() {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .bufferSync().get();

    this.mseHistory.unshift(mse);
  }

  updateLearningRate() {
    if (this.mseHistory.length < 2) {
      return;
    }

    if (this.mseHistory[0] > this.mseHistory[1]) {
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

      this.recordMSE();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    return this.processFeatures(observations).matMul(this.weights);
  }

  test(testFeatures, testLabels) {
    testFeatures = this.processFeatures(testFeatures)
    testLabels = tf.tensor(testLabels);

    const predictions = testFeatures.matMul(this.weights);

    const res = testLabels.sub(predictions).pow(2).sum()
      .bufferSync().get();

    const tot = testLabels.sub(testLabels.mean()).pow(2).sum()
      .bufferSync().get();
    
    return 1 - res / tot;
  }

}

module.exports = LinearRegression;