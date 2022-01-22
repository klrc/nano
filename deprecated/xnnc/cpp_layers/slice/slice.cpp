#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include "slice.hpp"

using namespace std;

void slice_copy(const std::vector<Tensor<float> *> &inputs, const std::vector<Tensor<float> *> &outputs, int axis,
                int start, int end) {
    TensorShape input_shape = inputs[0]->getShape();
    int chunk_offset = input_shape[axis];
    int chunk_size = end - start;
    int num_chunks = 1;
    // get total numbers of chunks
    for (int i = 0; i < axis; i++) {
        num_chunks *= input_shape[i];
    }
    // get chunk size
    for (int i = axis + 1; i < 4; i++) {
        chunk_offset *= input_shape[i];
        chunk_size *= input_shape[i];
    }
    // copy memory
    float *input_memory = inputs[0]->getMutableData();
    float *output_memory = outputs[0]->getMutableData();
    for (int i = 0; i < num_chunks; i++) {
        for (int j = 0; j < chunk_size; j++) {
            output_memory[i * chunk_size + j] = input_memory[i * chunk_offset + j];
        }
    }
}


class CustomSliceLayer : public CppCustomLayer {
   public:
    int axis;
    int start;
    int end;

    ~CustomSliceLayer() { /* release codes here */
    }

    explicit CustomSliceLayer(int axis, int start, int end) {
        /* initialize codes here */
        this->axis = axis;
        this->start = start;
        this->end = end;
    }

    virtual void reshape(const std::vector<Tensor<float> *> &inputs, const std::vector<Tensor<float> *> &outputs) {
        TensorShape top_shape(4);
        top_shape[this->axis] = this->end - this->start;
        outputs[0]->reshape(top_shape);
    }

    virtual void forward(const std::vector<Tensor<float> *> &inputs, const std::vector<Tensor<float> *> &outputs) {
        slice_copy(inputs, outputs, axis, start, end);
    }

    virtual string getTypeName() const {
        return string("slice");
    }

    virtual void getCfunctionDescr(CFunctionDescr &descr) {
        descr.setName("xi_slice");
        // TODO: support nx inputs here
        descr.addInputLayout(MEMORY_LAYOUT_XI_WHDN);
        descr.addOutputLayout(MEMORY_LAYOUT_XI_WHDN);
    }

   private:
};

extern "C" {
XNNC_EXPORT CppCustomLayer *createCppCustomLayer(map<string, string> & param_map) {
    CustomSliceLayer *layer;
    /* initialize parameters here */
    int axis = strtol(param_map["axis"].c_str(), NULL, 0);
    int start = strtol(param_map["start"].c_str(), NULL, 0);
    int end = strtol(param_map["end"].c_str(), NULL, 0);
    layer = new CustomSliceLayer(axis, start, end);
    return layer;
}
}