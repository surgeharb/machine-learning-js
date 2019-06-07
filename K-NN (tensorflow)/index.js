require('@tensorflow/tfjs-node-gpu');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

(function run() {
  console.time();

  const options = {
    shuffle: true,
    splitTest: 10,
    dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living'],
    labelColumns: ['price']
  };

  let { features, labels, testFeatures, testLabels } = loadCSV(__dirname + '/kc_house_data.csv', options);

  features = tf.tensor(features);
  labels = tf.tensor(labels);

  testFeatures.forEach((testPoint, i) => {
    const result = knn(features, labels, tf.tensor(testPoint), 10);
    // console.log('Guess', result, testPoint[0], '$');

    const err = (testLabels[i][0] - result) / testLabels[i][0];
    console.log('Error:', Math.round(err * 100 * 100) / 100, '%');
  })

  console.log();
  console.timeEnd();
})();

function knn(features, labels, predictionPoint, k) {
  const { mean, variance } = tf.moments(features, 0);

  const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(.5));

  return (
    features
      .sub(mean)
      .div(variance.pow(.5))
      .sub(scaledPrediction)
      .pow(2)
      .sum(1)
      .pow(0.5)
      .expandDims(1)
      .concat(labels, 1)
      .unstack()
      .sort((a, b) => a.bufferSync().get(0) - b.bufferSync().get(0))
      .slice(0, k)
      .reduce((acc, pair) => acc + pair.bufferSync().get(1), 0) / k
  );
}