close all; clc;

XTrain = chord;
%this is regression
numClasses = 1;

load FCWeights
veclen = length(chord);
layers = [ ...
    imageInputLayer([veclen 1],'Name','input','mean',0)
    %sequenceInputLayer(veclen,'name','input')
    %flattenLayer('name','flat')
    fullyConnectedLayer(veclen,'name','hidden','Weights',ones(veclen, veclen), ...
    'Bias',ones(veclen,1))
    %regressionLayer
    acousticIntermediateLayer("Speaker_Mic", 8192,chord)];
lgraph = layerGraph(layers);
dlnet = assembleNetwork(layers);

%%
numEpochs = 2;
miniBatchSize = veclen;

plots = "training-progress";
executionEnvironment = "auto";

if plots == "training-progress"
    figure
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on
end

numObservations = veclen;%numel(YTrain);
numIterationsPerEpoch = floor(numObservations./miniBatchSize);

iteration = 0;
start = tic;

% Loop over epochs. Largest scope of data - whole set/signal
for epoch = 1:numEpochs
    
    % Shuffle data. Not needed here
    %idx = randperm(numel(YTrain));
    %XTrain = XTrain(:,:,:,idx);
    %YTrain = YTrain(idx);
    
    % Loop over mini-batches.
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        
        % Read mini-batch of data and convert the labels to dummy
        % variables.
        %idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
         X = XTrain';
        
        %Y = zeros(numClasses, miniBatchSize, 'single');
        %for c = 1:numClasses
        %    Y(c,YTrain(idx)==classes(c)) = 1;
        %end
        Y = X;
        % Convert mini-batch of data to dlarray.
        dlX = dlarray(single(X));
        
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end
        
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients,state,loss] = dlfeval(@modelGradients,dlnet,dlX,Y);
        dlnet.State = state;
        
        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate/(1 + decay*iteration);
        
        % Update the network parameters using the SGDM optimizer.
        [dlnet, velocity] = sgdmupdate(dlnet, gradients, velocity, learnRate, momentum);
        
        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
end



function [gradients,state,loss] = modelGradients(dlnet,dlX,Y)

[dlYPred,state] = forward(dlnet,dlX);

loss = mse(dlYPred,Y);
gradients = dlgradient(loss,dlnet.Learnables);

end