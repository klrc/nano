#include "yoloxpp.hpp"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

using namespace std;

#define CONFIDENCE_THRESH 0.25
#define IOU_THRESH 0.45
#define MAX_DETECTIONS 300
#define MAX_NMS 30000  // maximum number of boxes into torchvision.ops.nms()

#define NUM_CLASSES 3
#define NUM_STRIDES 3

#define INPUT_HEIGHT 256
#define INPUT_WIDTH 512

typedef struct BoundingBox {
    int label;
    float score;
    float x1;
    float y1;
    float x2;
    float y2;
} BoundingBox;

bool cmp_bbox(BoundingBox& a, BoundingBox& b) {
    return a.score > b.score;  // descending order
}

inline float fast_exp(float x) {
    union Data {
        uint32_t i;
        float f;
    };  // v{};
    union Data v;
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

inline float clip(float x, float _min, float _max) {
    if (x < _min) return _min;
    if (x > _max) return _max;
    return x;
}

inline float detection_iou(const BoundingBox box1, const BoundingBox box2) {
    float area_a = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    float area_b = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    float w = min(box1.x2, box2.x2) - max(box1.x1, box2.x1);
    float h = min(box1.y2, box2.y2) - max(box1.y1, box2.y1);
    if (w <= 0 or h <= 0) return 0;
    float area_c = w * h;
    return area_c / (area_a + area_b - area_c);
}

void detection_nms(vector<BoundingBox>& candidates) {
    // Runs Non-Maximum Suppression (NMS) on inference results
    // sort by confidence
    sort(candidates.begin(), candidates.end(), cmp_bbox);
    if (candidates.size() > MAX_NMS) {
        candidates.assign(candidates.begin(), candidates.begin() + MAX_NMS);
    }
    // Non-Max Suppression
    for (int i = 0; i < int(candidates.size()); i++) {
        for (int j = i + 1; j < int(candidates.size());) {
            if (detection_iou(candidates[i], candidates[j]) >= IOU_THRESH) {
                candidates.erase(candidates.begin() + j);
            } else {
                j++;
            }
        }
    }
}

inline float tensor_by_index(float* tensor, int n, int c, int h, int w, const int max_n, const int max_c,
                             const int max_h) {
    // read data in WHDN order
    return tensor[max_n * (max_c * (max_h * w + h) + c) + n];
}

void check_dense_anchors(Tensor<float>* input, const int stride, vector<BoundingBox> results) {
    vector<float> cls_stack;
    float* tensor = input->getMutableData();
    const int IN_D = input->getDepth();
    const int IN_H = input->getHeight();
    const int IN_W = input->getWidth();
    // currently support only N=1
    for (int h = 0; h < IN_H; h++) {
        for (int w = 0; w < IN_W; w++) {
            // find anchor with confidence > threshold
            cls_stack.clear();
            for (int c = 4; c < 4 + NUM_CLASSES; c++) {
                cls_stack.push_back(tensor_by_index(tensor, 0, c, h, w, 1, IN_D, IN_H));
            }
            int label = max_element(cls_stack.begin(), cls_stack.end()) - cls_stack.begin();  // 0, 1, 2, ...
            float confidence = sigmoid(cls_stack[label]);
            // collect data
            if (confidence > CONFIDENCE_THRESH) {
                // get real-size x, y, w, h, class label, class score
                float x1 = sigmoid(tensor_by_index(tensor, 0, 0, h, w, 1, IN_D, IN_H)) * 4 * stride;
                float y1 = sigmoid(tensor_by_index(tensor, 0, 1, h, w, 1, IN_D, IN_H)) * 4 * stride;
                float x2 = sigmoid(tensor_by_index(tensor, 0, 2, h, w, 1, IN_D, IN_H)) * 4 * stride;
                float y2 = sigmoid(tensor_by_index(tensor, 0, 3, h, w, 1, IN_D, IN_H)) * 4 * stride;
                // Box (center x, center y, width, height) to (x1, y1, x2, y2)
                BoundingBox obj;
                // scale to real size
                obj.x1 = clip(((w + 0.5) * stride - x1) / INPUT_WIDTH, 0, 1);
                obj.y1 = clip(((h + 0.5) * stride - y1) / INPUT_HEIGHT, 0, 1);
                obj.x2 = clip(((w + 0.5) * stride + x2) / INPUT_WIDTH, 0, 1);
                obj.y2 = clip(((h + 0.5) * stride + y2) / INPUT_HEIGHT, 0, 1);
                obj.score = confidence;
                obj.label = label;
                results.push_back(obj);
            }
        }
    }
}

void post_process(const std::vector<Tensor<float>*>& inputs, const std::vector<Tensor<float>*>& outputs) {
    float* output_memory = outputs[0]->getMutableData();
    vector<BoundingBox> results;
    const int strides[] = {8, 16, 32, 64};
    const int output_labels[] = {15, 2, 7};  // background-0, person-15, bike-2, car-7

    // select cadidates with obj_score > conf_thresh
    for (int i = 0; i < NUM_STRIDES; i++) {
        check_dense_anchors(inputs[i], strides[i], results);
    }

    // run nms
    detection_nms(results);

    // memset top data
    for (int i = 0; i < MAX_DETECTIONS * 6; i++) {
        output_memory[i] = 0.0f;
    }

    for (int i = 0; i < results.size(); i++) {
        printf("[%d], %d, %f, %f, %f, %f, %f \n", i, output_labels[results[i].label], results[i].score, results[i].x1,
               results[i].y1, results[i].x2, results[i].y2);
        output_memory[i * 6 + 0] = output_labels[results[i].label];
        output_memory[i * 6 + 1] = results[i].score;
        output_memory[i * 6 + 2] = results[i].x1;
        output_memory[i * 6 + 3] = results[i].y1;
        output_memory[i * 6 + 4] = results[i].x2;
        output_memory[i * 6 + 5] = results[i].y2;
    }
}

class YOLOXPostProcessLayer : public CppCustomLayer {
   public:
    ~YOLOXPostProcessLayer() {
        /* release codes here */
    }

    explicit YOLOXPostProcessLayer() {
        /* initialize codes here */
    }

    virtual void reshape(const std::vector<Tensor<float>*>& inputs, const std::vector<Tensor<float>*>& outputs) {
        TensorShape top_shape(4);
        top_shape[0] = 1;
        top_shape[1] = MAX_DETECTIONS;
        top_shape[2] = 6;  // label, score, xyxy
        top_shape[3] = 1;
        outputs[0]->reshape(top_shape);
    }

    virtual void forward(const std::vector<Tensor<float>*>& inputs, const std::vector<Tensor<float>*>& outputs) {
        post_process(inputs, outputs);
    }

    virtual string getTypeName() const {
        return string("yoloxpp");
    }

    virtual void getCfunctionDescr(CFunctionDescr& descr) {
        descr.setName("xi_yoloxpp");
        // TODO: support nx inputs here
        for (int i = 0; i < NUM_STRIDES; i++) {
            descr.addInputLayout(MEMORY_LAYOUT_XI_WHDN);
        }
        descr.addOutputLayout(MEMORY_LAYOUT_XI_WHDN);
    }

   private:
};

extern "C" {
XNNC_EXPORT CppCustomLayer* createCppCustomLayer(map<string, string>& param_map) {
    YOLOXPostProcessLayer* layer;
    /* initialize parameters here */
    layer = new YOLOXPostProcessLayer();
    return layer;
}
}