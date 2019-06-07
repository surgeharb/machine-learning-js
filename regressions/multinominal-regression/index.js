require('@tensorflow/tfjs-node-gpu');
const tf = require('@tensorflow/tfjs');

const _ = require('lodash');
const mnist = require('mnist-data');
const plot = require('node-remote-plot');
const LogisticRegression = require('./logistic-regression');

const TRAINING_COUNT = 60000;
const TESTING_COUNT = 10000;

const encode = mnistData => {
  const features = mnistData.images.values.map(image => _.flatMap(image));

  const labels = mnistData.labels.values.map(label => {
    const row = Array(10).fill(0);
    row[label] = 1;
    return row;
  });

  return { features, labels };
};

const loadData = () => {
  const mnistTrainingData = mnist.training(0, TRAINING_COUNT);
  return encode(mnistTrainingData);
};

let encodedTrainingData = loadData();

const options = {
  learningRate: 1,
  iterations: 40,
  batchSize: 500
};

const regression = new LogisticRegression(
  encodedTrainingData.features,
  encodedTrainingData.labels,
  options
);

regression.train();

const mnistTestingData = mnist.testing(0, TESTING_COUNT);
const { features, labels } = encode(mnistTestingData);
const accuracy = regression.test(features, labels);
console.log(`Accuracy is: ${accuracy * 100}%`);

plot({
  x: regression.costHistory.reverse(),
});