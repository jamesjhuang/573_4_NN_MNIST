% CSE573 HW4
% James J. Huang

%% E.C.

trainImg = fopen('MNIST_data/train-images-idx3-ubyte','r','b');
MagicNumber = fread(trainImg, 1, 'int32');
nImages = fread(trainImg, 1, 'int32');
nRows = fread(trainImg, 1, 'int32');
nCols = fread(trainImg, 1, 'int32');

trainlbl = fopen('MNIST_data/train-labels-idx1-ubyte', 'r', 'b');
MagicNumber2 = fread(trainlbl, 1, 'int32');
nLabels = fread(trainlbl, 1, 'int32');

numImg = 5000;

realTrainImages = cell(numImg, 1);
trainLabels = zeros(numImg, 1);
tTrain = zeros(10, numImg);

for i = 1:numImg
    fseek(trainImg, 16 + (i-1) * 28 * 28, 'bof');
    img = fread(trainImg, 28*28, 'uchar');
    img = reshape(img, [28 28])';
    smallImg = img(5:24, 5:24);
    realTrainImages{i} = smallImg ./ 255;
    
    fseek(trainlbl, 8 + (i - 1), 'bof');
    trainLabels(i) = fread(trainlbl, 1, 'uchar');
    tTrain(trainLabels(i) + 1, i) = 1;
end

for i = 1:20
    subplot(4,5,i);
    imshow(realTrainImages{i});
end
%%
% Random generator seed
rng('default');

% Size of hidden layer
hiddenSize1 = 200;

% Train
autoenc1 = trainAutoencoder(realTrainImages', hiddenSize1, ...
    'MaxEpochs', 400, ...
    'L2WeightRegularization', 0.004, ...
    'SparsityRegularization', 4, ...
    'SparsityProportion', 0.15, ...
    'ScaleData', false);

view(autoenc1);

% Visualizing weights
figure()
plotWeights(autoenc1);

feat1 = encode(autoenc1, realTrainImages');

% Train 2nd autoencoder
hiddenSize2 = 100;
autoenc2 = trainAutoencoder(feat1, hiddenSize2, ...
    'MaxEpochs', 100, ...
    'L2WeightRegularization', 0.002, ...
    'SparsityRegularization', 4, ...
    'SparsityProportion', 0.1, ...
    'ScaleData', false);

view(autoenc2)

feat2 = encode(autoenc2, feat1);

softnet = trainSoftmaxLayer(feat2, tTrain, 'MaxEpochs', 400);

view(softnet);

% Forming deep network
deepnet = stack(autoenc1, autoenc2, softnet);
view(deepnet)

%% Testing NN
% Get number pixels
imgW = 20;
imgH = 20;
inputSize = imgW * imgH;

% Loading test images
testImg = fopen('MNIST_data/t10k-images-idx3-ubyte','r','b');
MagicNumber = fread(testImg, 1, 'int32');
nImages = fread(testImg, 1, 'int32');
nRows = fread(testImg, 1, 'int32');
nCols = fread(testImg, 1, 'int32');

testlbl = fopen('MNIST_data/t10k-labels-idx1-ubyte', 'r', 'b');
MagicNumber2 = fread(testlbl, 1, 'int32');
nLabels = fread(testlbl, 1, 'int32');

numImg = 5000;

realTestImages = cell(numImg, 1);
testLabels = zeros(numImg, 1);
tTest = zeros(10, numImg);

for i = 1:numImg
    fseek(testImg, 16 + (i-1) * 28 * 28, 'bof');
    img = fread(testImg, 28*28, 'uchar');
    img = reshape(img, [28 28])';
    smallImg = img(5:24, 5:24);
    realTestImages{i} = smallImg ./ 255;
    
    fseek(testlbl, 8 + (i - 1), 'bof');
    testLabels(i) = fread(testlbl, 1, 'uchar');
    tTest(testLabels(i) + 1, i) = 1;
end

% Vectorizing and storing
xTest = zeros(inputSize, numel(realTestImages));
for i = 1:numel(realTestImages)
    xTest(:, i) = realTestImages{i}(:);
end

y = deepnet(xTest);
plotconfusion(tTest, y);