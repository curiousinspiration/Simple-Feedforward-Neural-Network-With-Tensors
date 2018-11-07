/*
 * Example tool training feed forward neural network on non-linear data
 *
 */

#include "neural/layers/linear_layer.h"
#include "neural/layers/relu_layer.h"
#include "neural/loss/squared_error_loss.h"

#include <glog/logging.h>

using namespace neural;
using namespace std;

float CalcAverage(const vector<float>& vals)
{
    float sum = 0.0;
    for (size_t i = 0; i < vals.size(); ++i)
    {
        sum += vals.at(i);
    }
    return sum / ((float)vals.size());
}

int main(int argc, char const *argv[])
{
    // Define dataset
    vector<TTensorPtr> inputs = {
        Tensor::New({1,2}, {1.0, 1.0}),
        Tensor::New({1,2}, {1.0, 0.0}),
        Tensor::New({1,2}, {0.0, 1.0}),
    };

    vector<float> outputs = {
        1.0,
        0.0,
        -1.0
    };

    // Define model

    // first linear layer is 2x2
    // 2 inputs, 2 outputs
    // Therefore needs 4 weights
    LinearLayer firstLinearLayer(
        Tensor::New({2,2}, {
            -0.5, 1.2,
            0.6, -0.8
        })
    );

    // Non-linear activation
    ReLULayer activationLayer;
    
    // second linear layer is 2x1
    // 2 inputs, 1 output
    // Therefore needs 2 weights
    LinearLayer secondLinearLayer(
        Tensor::New({2,1}, {
            -0.5,
            1.2
        })
    );

    // Error function
    SquaredErrorLoss loss;

    // Training loop
    float learningRate = 0.1;
    size_t numEpochs = 100;
    for (size_t i = 0; i < numEpochs; ++i)
    {
        LOG(INFO) << "--EPOCH (" << i << ")--" << endl;
        vector<float> errorAcc;
        for (size_t j = 0; j < inputs.size(); ++j)
        {
            LOG(INFO) << "--ITER (" << i << "," << j << ")--" << endl;
            // Get training example
            TTensorPtr input = inputs.at(j);
            float targetOutput = outputs.at(j);

            // Forward pass
            TTensorPtr output0 = firstLinearLayer.Forward(input);
            TTensorPtr output1 = activationLayer.Forward(output0);
            TTensorPtr y_pred = secondLinearLayer.Forward(output1);
            float yPredVal = y_pred->At({0,0});
            LOG(INFO) << "Got prediction: " << yPredVal << " for target " << targetOutput << endl;

            // Calc Error
            float error = loss.Forward(yPredVal, targetOutput);
            errorAcc.push_back(error);
            LOG(INFO) << "Calculated error: " << error << endl;

            // Backward pass
            float errorGrad = loss.Backward(yPredVal, targetOutput);
            TTensorPtr y_predGrad = secondLinearLayer.Backward(output1, Tensor::New({1,1}, {errorGrad}));
            TTensorPtr grad1 = activationLayer.Backward(output0, y_predGrad);
            TTensorPtr grad0 = firstLinearLayer.Backward(input, grad1);
        }

        // Compute average error
        float avgError = CalcAverage(errorAcc);
        LOG(INFO) << "avgError = " << avgError << endl;

        // Gradient Descent
        secondLinearLayer.UpdateWeights(learningRate);
        firstLinearLayer.UpdateWeights(learningRate);
    }

    return 0;
}