classdef acousticLayer < nnet.layer.RegressionLayer
        
    properties
        % (Optional) Layer properties.

        % Layer properties go here.
            output;
            loss;
            Fs;
    end
 
    methods
        function layer = acousticLayer(name, Fs)           

            % Layer constructor function goes here.
            % Set layer name.
            layer.Name = name;
            layer.Fs = Fs;

            % Set layer description.
            layer.Description = 'Plays an input sequence through a speaker and calculates loss from it';
        end

        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the MAE loss between
            % the predictions Y and the training targets T.
            
            %newY = reshape(Y, numel(Y),1);

            truePredictions = playVectorThroughSpeaker(Y, layer.Fs);
            layer.output = truePredictions;
            %dlarray(truePredictions);
            
            % Calculate MAE.
            % R = size(trueOutput,1);
            %size(T);
            %size(truePredictions);
            meanSquaredError = sum((truePredictions-T).^2)/length(T);
            layer.loss = meanSquaredError;
            loss = meanSquaredError
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % (Optional) Backward propagate the derivative of the loss 
            % function.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         dLdY  - Derivative of the loss with respect to the 
            %                 predictions Y        

            % Layer backward loss function goes here.
            size(layer.output)
            size(Y)
            dLdY = layer.loss - mean(layer.loss - Y);            
            %dLdY = forwardLoss(layer, Y, T) - mean(forwardLoss(layer, Y, T) - Y);            
        end
    end
end