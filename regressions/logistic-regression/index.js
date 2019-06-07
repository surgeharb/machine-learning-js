require('@tensorflow/tfjs-node-gpu');
const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');

const { join } = require('path');
const loadCSV = require('../load-csv');
const LogisticRegression = require('./logistic-regression');

const options = {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['displacement', 'horsepower', 'weight'],
  labelColumns: ['passedemissions'],
  converters: {
    passedemissions: value => {
      return value === 'TRUE' ? 1 : 0;
    }
  }
};

let { features, labels, testFeatures, testLabels } = loadCSV(join(__dirname, '../data/cars.csv'), options);

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 50,
});

regression.train();

const accuracy = regression.test(testFeatures, testLabels);
console.log('Accuracy:', `${accuracy * 100}%`);

plot({
  x: regression.costHistory.reverse()
})