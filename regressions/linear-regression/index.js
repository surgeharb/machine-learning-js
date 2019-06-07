require('@tensorflow/tfjs-node-gpu');
const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');

const { join } = require('path');
const loadCSV = require('../load-csv');
const LinearRegression = require('./linear-regression');

const options = {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'weight', 'displacement', 'cylinders'],
  labelColumns: ['mpg']
};

let { features, labels, testFeatures, testLabels } = loadCSV(join(__dirname, '../data/cars.csv'), options);

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 3,
  batchSize: 10,
});

regression.train();

const r2 = regression.test(testFeatures, testLabels);
console.log('r2:', r2);

plot({
  x: regression.mseHistory.reverse(),
  xLabel: 'Iteration #',
  yLabel: 'Mean Squared Error'
});

console.log('Prediction:');

regression.predict([
  [155, 1.7, 122, 4],
  [68, 0.8, 60, 3],
]).print();
