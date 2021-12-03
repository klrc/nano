#include <cstddef>
#include <string>
#include <vector>

#include "xnnc/cpp_custom_layer.hpp"
using namespace xnnc;
using namespace std;

void copy_mem(float* input_memory, float* outout_memory, int start_index, int offset) {
    for (int i = start_index; i < start_index + offset; i++) {
        outout_memory[i] = input_memory[i];
    }
}

class SliceLayer : public CppCustomLayer {
   public:
    int dim;
    int chunks;
    ~SliceLayer() {
        /* release codes here */
    }

    explicit SliceLayer(int chunks, int dim) {
        // performs uniform decomposition only
        this->dim = dim;
        this->chunks = chunks;
    }

    virtual void reshape(const std::vector<Tensor<float>*>& inputs, const std::vector<Tensor<float>*>& outputs) {
        TensorShape input_shape = inputs[0]->getShape();
        for (int i = 0; i < chunks; i++) {
            TensorShape top_shape(4);
            for (int dim = 0; dim < 4; dim++) {
                if (dim != dim) {
                    top_shape[dim] = input_shape[dim];
                } else {
                    top_shape[dim] = input_shape[dim] / chunks;
                }
            }
            outputs[i]->reshape(top_shape);
        }
    }

    virtual void forward(const std::vector<Tensor<float>*>& inputs, const std::vector<Tensor<float>*>& outputs) {
        TensorShape input_shape = inputs[0]->getShape();
        float* input_memory = inputs[0]->getMutableData();
        // get size for each chunk
        int total_size = 1;
        int chunk_size = input_shape[dim] / chunks;
        for (int i = 0; i < 4; i++) {
            total_size *= input_shape[i];
            if (i > dim) {
                chunk_size *= input_shape[i];
            }
        }
        // allocate by chunk size in turn
        int allocated_size = 0;
        while (allocated_size < total_size) {
            for (int i = 0; i < chunks; i++) {
                float* output_memory = outputs[i]->getMutableData();
                copy_mem(input_memory, output_memory, allocated_size, chunk_size);
                allocated_size += chunk_size;
            }
        }
    }

    virtual string getTypeName() const {
        return string("slice");
    }

    virtual void getCfunctionDescr(CFunctionDescr& descr) {
        descr.setName("xi_slice");
        descr.addInputLayout(MEMORY_LAYOUT_XI_WHDN);
        for (int i = 0; i < chunks; i++) {
            descr.addOutputLayout(MEMORY_LAYOUT_XI_WHDN);
        }
    }

   private:
};

extern "C" {
XNNC_EXPORT CppCustomLayer* createCppCustomLayer(map<string, string>& param_map) {
    SliceLayer* layer;
    /* initialize parameters here */
    int chunks = strtol(param_map["chunks"].c_str(), NULL, 0);
    int dim = 1;
    if (param_map.count("dim") > 0) {
        dim = strtol(param_map["dim"].c_str(), NULL, 0);
    }
    layer = new SliceLayer(chunks, dim);
    return layer;
}
}