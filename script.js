import {MnistData} from './data.js';

const getData = getDataFunction;
const createModel =  createModelFunction;
const trainModel = trainModelFunction;
const displayData = displayDataFunction;
const evaluateModel = evaluateModelFunction;

const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

async function run() {  
    const data = await getData();
  
    await displayDataFunction(data, 30);
  
    const model = createModel();
    tfvis.show.modelSummary({name: 'Model Architecture'}, model);
  
    await trainModel(model, data, 20);
  
    await evaluateModel(model, data);
}

/**
  * @desc retrieves data from defined location
  * @return wine data as json
*/
async function getDataFunction() {
    var data = new MnistData();
    await data.load();
    return data;
}

async function singleImagePlot(image)
{
  const canvas = document.createElement('canvas');
  canvas.width = 28;
  canvas.height = 28;
  canvas.style = 'margin: 4px;';
  await tf.browser.toPixels(image, canvas);
  return canvas;
}

async function displayDataFunction(data, numOfImages = 10) {
    
  const inputDataSurface =
    tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});  

  const examples = data.nextDataBatch(numOfImages, true);
  
  for (let i = 0; i < numOfImages; i++) {
    const image = tf.tidy(() => {
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });
    
    const canvas = await singleImagePlot(image)
    inputDataSurface.drawArea.appendChild(canvas);

    image.dispose();
  }
}

function createModelFunction() {
  const cnn = tf.sequential();
  
  cnn.add(tf.layers.conv2d({
      inputShape: [28, 28, 1],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
  cnn.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

  cnn.add(tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
  cnn.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

  cnn.add(tf.layers.flatten());

  cnn.add(tf.layers.dense({
    units: 10,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));
  
  cnn.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return cnn;
}

function getBatch(data, size, test = false)
{
    return tf.tidy(() => {
        const d = data.nextDataBatch(size, test);
        return [
          d.xs.reshape([size, 28, 28, 1]),
          d.labels
        ];
      });
}

async function trainModelFunction(model, data, epochs) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
      name: 'Model Training', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
    
    const batchSize = 512;
  
    const [trainX, trainY] = getBatch(data, 5500);
    const [testX, testY] = getBatch(data, 1000, true);
  
    return model.fit(trainX, trainY, {
      batchSize: batchSize,
      validationData: [testX, testY],
      epochs: epochs,
      shuffle: true,
      callbacks: fitCallbacks
    });
}


function predict(model, data, testDataSize = 500) {
  const testData = data.nextDataBatch(testDataSize, true);
  const testxs = testData.xs.reshape([testDataSize, 28, 28, 1]);
  const labels = testData.labels.argMax([-1]);
  const preds = model.predict(testxs).argMax([-1]);

  testxs.dispose();
  return [preds, labels];
}


async function displayAccuracyPerClass(model, data) {
  const [preds, labels] = predict(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = {name: 'Accuracy', tab: 'Evaluation'};
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function displayConfusionMatrix(model, data) {
  const [preds, labels] = predict(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
  tfvis.render.confusionMatrix(
      container, {values: confusionMatrix}, classNames);

  labels.dispose();
}

async function evaluateModelFunction(model, data)
{
  await displayAccuracyPerClass(model, data);
  await displayConfusionMatrix(model, data);
}

document.addEventListener('DOMContentLoaded', run);