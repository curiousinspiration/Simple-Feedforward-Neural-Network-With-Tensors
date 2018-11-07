/*
 * Linear Layer Implementation
 *
 */

#include "neural/layers/linear_layer.h"
#include "neural/math/tensor_math.h"

#include <glog/logging.h>

#include <sstream>

using namespace std;

namespace neural
{

LinearLayer::LinearLayer(const TTensorPtr& a_weights, bool a_hasBias)
    : m_hasBias(a_hasBias)
    , m_weights(a_weights->ToMutable()) // weights are learnable, hence mutable
{
    // if there is a bias, add an extra row to the weights
    if (m_hasBias)
    {
        m_weights = TensorMath::AddRow(m_weights, 1.0)->ToMutable();
    }
}

TTensorPtr LinearLayer::Forward(const TTensorPtr& a_input) const
{
    // Make a local copy so we can add bias if needed
    TTensorPtr l_input = a_input;

    if (m_hasBias)
    {
        // add an extra column of 1s
        l_input = TensorMath::AddCol(l_input, 1.0);
    }

    return TensorMath::Multiply(l_input, m_weights);
}

TTensorPtr LinearLayer::Backward(const TTensorPtr& a_origInput, const TTensorPtr& a_gradInput)
{
    // orig input might have had bias
    TTensorPtr l_input = a_origInput;
    if (m_hasBias)
    {
        // add an extra column of 1s
        l_input = TensorMath::AddCol(l_input, 1.0);
    }

    // Gradient wrt weights
    TTensorPtr gradWrtWeights = TensorMath::Multiply(TensorMath::Transpose(l_input), a_gradInput);
    m_weightGrads.push_back(gradWrtWeights);

    // Gradient with respect to output will be (grad from next layer) * weights
    // Weights is the gradient of the output with the respect to the input for this layer and we multipy because of the chain rule
    TTensorPtr gradWrtOutput = TensorMath::Multiply(a_gradInput, TensorMath::Transpose(m_weights));

    if (m_hasBias)
    {
        gradWrtOutput = TensorMath::RemoveCol(gradWrtOutput);
    }

    return gradWrtOutput;
}

void LinearLayer::UpdateWeights(float a_learningRate)
{
    TTensorPtr l_gradient = CalcAvgWeightGrad();

    for (size_t i = 0; i < m_weights->Shape().at(0); ++i)
    {
        for (size_t j = 0; j < m_weights->Shape().at(1); ++j)
        {
            float l_newVal = m_weights->At({i,j}) - (a_learningRate * l_gradient->At({i,j}));
            m_weights->SetAt({i,j},l_newVal);
        }
    }

    // clear gradients
    m_weightGrads.clear();
}

TTensorPtr LinearLayer::CalcAvgWeightGrad() const
{
    // Init with zeros
    TMutableTensorPtr average = Tensor::Zeros(m_weightGrads.at(0)->Shape());

    // Sum up
    for (const TTensorPtr& grad : m_weightGrads)
    {
        for (size_t i = 0; i < grad->Shape().at(0); ++i)
        {
            for (size_t j = 0; j < grad->Shape().at(1); ++j)
            {
                float l_sum = average->At({i, j}) + grad->At({i,j});
                average->SetAt({i,j}, l_sum);
            }
        }
    }

    // Average
    float numGrads = (float)m_weightGrads.size();
    for (size_t i = 0; i < m_weightGrads.at(0)->Shape().at(0); ++i)
    {
        for (size_t j = 0; j < m_weightGrads.at(0)->Shape().at(1); ++j)
        {
            float l_avg = average->At({i,j}) / numGrads;
            average->SetAt({i,j}, l_avg);
        }
    }

    return average;
}

} // namespace neural
