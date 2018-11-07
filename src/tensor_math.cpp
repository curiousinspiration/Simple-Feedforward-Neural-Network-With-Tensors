/* src/tensor_math.cpp
 *
 * Tensor Math Implementation
 *
 */

#include "neural/math/tensor_math.h"

#include <glog/logging.h>
  
#include <sstream>

using namespace std;

namespace neural
{

TTensorPtr TensorMath::Multiply(const TTensorPtr& a_lhs, const TTensorPtr& a_rhs)
{
    if (a_lhs->Shape().size() != 2 || a_rhs->Shape().size() != 2)
    {
        stringstream l_ss;
        l_ss << "TensorMath::Multiply for tensors of shape.size() != 2 is not supported. "
             << "a_lhs.size = " << a_lhs->Shape().size()
             << "a_rhs.size = " << a_rhs->Shape().size()
             << endl;
        LOG(ERROR) << l_ss.str() << endl;
        throw(runtime_error(l_ss.str()));
    }

    // Check to make sure the inner dimensions of our matrices line up
    if (a_lhs->Shape().at(1) != a_rhs->Shape().at(0))
    {
        stringstream l_ss;
        l_ss << "TensorMath::Multiply Inner dimensions of matrices must match "
             << a_lhs->Shape().at(1) << " != " << a_rhs->Shape().at(0);

        LOG(ERROR) << l_ss.str() << endl;
        throw(runtime_error(l_ss.str()));
    }

    // initialize our return matrix with the correct shape,
    // ie the outer sizes of our inputs and rhs
    TMutableTensorPtr l_ret = Tensor::Zeros({a_lhs->Shape().at(0), a_rhs->Shape().at(1)});

    // walk over rows of lhs
    for (size_t i = 0; i < a_lhs->Shape().at(0); ++i)
    {
        // walk over cols of rhs
        for (size_t j = 0; j < a_rhs->Shape().at(1); ++j)
        {
            // multiply element from row i in lhs by col j in weight
            // and sum up the values into i,j
            for (size_t k = 0; k < a_rhs->Shape().at(0); ++k)
            {
                // LOG(INFO) << "i " << i << " j " << j << " k " << a_lhs.at(i).at(k) << " * " << a_rhs.at(k).at(j) << endl;
                // l_ret.at(i).at(j) += (a_lhs.at(i).at(k) * a_rhs.at(k).at(j));
                float l_val = l_ret->At({i,j}) + (a_lhs->At({i,k}) * a_rhs->At({k,j}));
                l_ret->SetAt({i,j}, l_val);
            }
        }
    }
    return l_ret;
}

TTensorPtr TensorMath::Transpose(const TTensorPtr& a_mat)
{
    if (a_mat->Shape().size() != 2)
    {
        throw(runtime_error("TensorMath::Transpose for tensors of shape > 2 is not supported yet."));
    }

    size_t x = a_mat->Shape().at(0);
    size_t y = a_mat->Shape().at(1);
    
    TMutableTensorPtr l_ret = Tensor::New({y, x});
    for (size_t i = 0; i < x; ++i)
    {
        for (size_t j = 0; j < y; ++j)
        {
            l_ret->SetAt({j, i}, a_mat->At({i, j}));
        }
    }

    return l_ret;
}

TTensorPtr TensorMath::AddCol(const TTensorPtr& a_tensor, float a_val)
{
    vector<size_t> l_shape = a_tensor->Shape();
    if (l_shape.size() != 2)
    {
        string l_error("TensorMath::AddCol cannot call add rol on non-matrix tensor");
        LOG(ERROR) << l_error << endl;
        throw(l_error);
    }

    // Add the column to shape
    l_shape.at(1) += 1; 

    // Copy out the old data
    vector<float> l_data = a_tensor->Data();

    // Iterate over rows
    for (size_t i = 0; i < l_shape.at(0); ++i)
    {
        // Get offset into data vector
        size_t l_offset = (i * l_shape.at(1)) + (l_shape.at(1) - 1);

        // If we are at the end of the data
        if (l_offset >= l_data.size())
        {
            l_data.push_back(a_val);
        }
        else
        {
            l_data.insert(l_data.begin() + l_offset, a_val);
        }
    }

    // Return a new tensor with the new shape & data
    return Tensor::New(l_shape, l_data);
}

TTensorPtr TensorMath::RemoveCol(const TTensorPtr& a_tensor)
{
    vector<size_t> l_shape = a_tensor->Shape();
    if (l_shape.size() != 2)
    {
        string l_error("TensorMath::RemoveCol cannot call add rol on non-matrix tensor");
        LOG(ERROR) << l_error << endl;
        throw(l_error);
    }

    // Copy out the old data
    vector<float> l_data = a_tensor->Data();

    // Iterate over rows, but start at the back to preserve indices
    // If we delete from the front, all the indices will be shifted
    // And it will not work
    for (int i = l_shape.at(0)-1; i >= 0; --i)
    {
        // Get offset into data vector
        int l_offset = (i * l_shape.at(1)) + (l_shape.at(1) - 1);
        l_data.erase(l_data.begin()+l_offset);
    }

    // Remove the column from the shape
    l_shape.at(1) -= 1;

    // Return a new tensor with the new shape & data
    return Tensor::New(l_shape, l_data);
}

TTensorPtr TensorMath::AddRow(const TTensorPtr& a_tensor, float a_val)
{
    vector<size_t> l_shape = a_tensor->Shape();
    if (l_shape.size() != 2)
    {
        string l_error("TensorMath::RemoveCol cannot call add rol on non-matrix tensor");
        LOG(ERROR) << l_error << endl;
        throw(l_error);
    }

    // Copy out the old data
    vector<float> l_data = a_tensor->Data();
    for (size_t i = 0; i < l_shape.at(1); ++i)
    {
        l_data.push_back(a_val);
    }
    l_shape.at(0) += 1;
    return Tensor::New(l_shape, l_data);
}

TTensorPtr TensorMath::RemoveRow(const TTensorPtr& a_tensor)
{
    vector<size_t> l_shape = a_tensor->Shape();
    if (l_shape.size() != 2)
    {
        string l_error("TensorMath::RemoveCol cannot call add rol on non-matrix tensor");
        LOG(ERROR) << l_error << endl;
        throw(l_error);
    }

    // Copy out the old data
    vector<float> l_data = a_tensor->Data();
    for (size_t i = 0; i < l_shape.at(1); ++i)
    {
        // remove from the back
        l_data.pop_back();
    }
    l_shape.at(0) -= 1;
    return Tensor::New(l_shape, l_data);
}

} // namespace neural
