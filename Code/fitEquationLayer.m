classdef fitEquationLayer < nnet.layer.Layer

    properties
        % (Optional) Layer properties.
            output;
            loss;
            Fs;
            compare;
    end
    
    methods
        function layer = fitEquationLayer(name, Fs, compare)
           
            % Layer constructor function goes here.
            % Set layer name.
            layer.Name = name;
            layer.Fs = Fs;
            layer.compare = compare;
            % Set layer description.
            layer.Description = 'Plays an input sequence through a speaker and calculates loss from it';
 
        end
        
        function Z = predict(layer, Y)
            t = class(Y)
            if isa(Y,'gpuArray')
                %this is the 'underlying class'
                ul = class(gather(Y));
            end
            Z = playVectorThroughSpeaker(Y, layer.Fs);
            
            Z = extractdata(Z);
            
            if isa(Y,'gpuArray')
                Z = cast(Z, ul);
                Z = gpuArray(Z);
            elseif ~isa(Y, 'dlarray')
                Z = cast(Z, t);
            else
                Z = dlarray(Z);
            end
                      
        end

        function [Z1, memory] = forward(layer, X1)
            t = class(X1);
            if isa(X1,'gpuArray')
                %this is the 'underlying class'
                ul = class(gather(X1));
            end
            % (Optional) Forward input data through the layer at training
            % time and output the result and a memory value.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            %         memory      - Memory value for custom backward propagation

            % Layer forward function for training goes here.
            
            
            % loss = forwardLoss(layer, Y, T) returns the MAE loss between
            % the predictions Y and the training targets T.
            
            %newY = reshape(Y, numel(Y),1);

            Z1 = playVectorThroughSpeaker(X1, layer.Fs);
           
            %memory = 0;
            %class(memory)
            
            Z1= extractdata(Z1);
            
            if isa(X1,'gpuArray')
                Z1 = cast(Z1, ul);
                Z1 = gpuArray(Z1);
            elseif ~isa(X1, 'dlarray')
                Z1 = cast(Z1, t);
            else
                Z1 = dlarray(Z1);
            end
                      
        end

        function dLdY = backward(layer, Z1,~,~,~)
            t = class(Z1);
            if isa(Z1,'gpuArray')
                %this is the 'underlying class'
                ul = class(gather(Z1));
            end
            % (Optional) Backward propagate the derivative of the loss  
            % function through the layer.
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X1, ..., Xn       - Input data
            %         Z1, ..., Zm       - Outputs of layer forward function            
            %         dLdZ1, ..., dLdZm - Gradients propagated from the next layers
            %         memory            - Memory value from forward function
            % Outputs:
            %         dLdX1, ..., dLdXn - Derivatives of the loss with respect to the
            %                             inputs
            %         dLdW1, ..., dLdWk - Derivatives of the loss with respect to each
            %                             learnable parameter
            
            % Layer backward function goes here.
            
                        
            comp = dlarray(layer.compare);
            meanSquaredError = sum((comp-Z1).^2)/length(comp);

            %dLdY = loss - mean(loss - Y);            
            dLdY = meanSquaredError - (comp - Z1);    
            if isa(dLdY,'dlarray')
                dLdY = extractdata(dLdY);
            elseif isa(Z1,'gpuArray')
                dLdY = gather(dLdY);
            end

            if isa(Z1,'gpuArray')
                dLdY = cast(dLdY, ul);
                dLdY = gpuArray(dLdY);
            elseif ~isa(Z1, 'dlarray')
                dLdY = cast(dLdY, t);
            else
                dLdY = dlarray(dLdY);
            end
        end
    end
end