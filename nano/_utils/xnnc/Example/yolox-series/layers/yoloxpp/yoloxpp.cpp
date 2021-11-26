#include "yoloxpp.hpp"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

using namespace std;

#define CONFIDENCE_THRESH 0.2
#define IOU_THRESH 0.45
#define MAX_NMS 30000  // maximum number of boxes into torchvision.ops.nms()
#define FEATURE_STAGES 3
#define MAX_DETECTIONS 300
#define INPUT_HEIGHT 224
#define INPUT_WIDTH 416

typedef struct BoundingBox {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoundingBox;

bool cmp_bbox(BoundingBox& a, BoundingBox& b) {
    return a.score > b.score;  // descending order
}

inline float clip(float x, float _min, float _max) {
    if (x < _min) return _min;
    if (x > _max) return _max;
    return x;
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

void post_process(const std::vector<Tensor<float>*>& inputs, const std::vector<Tensor<float>*>& outputs) {
    float* output_memory = outputs[0]->getMutableData();
    vector<BoundingBox> results;
    vector<float> cls_stack;

    const int strides[] = {8, 16, 32};
    const float anchors[] = {11.3359375, 13.6875, 30.359375, 46.34375, 143.0, 129.5};

    // select cadidates with obj_score > conf_thresh
    for (int i = 0; i < FEATURE_STAGES; i++) {
        float* input_memory = inputs[i]->getMutableData();
        const int IN_H = inputs[i]->getDepth();
        const int IN_W = inputs[i]->getHeight();
        const int IN_C = inputs[i]->getWidth();
        for (int h = 0; h < IN_H; h++) {
            for (int w = 0; w < IN_W; w++) {
                int cursor = (h * IN_W + w) * IN_C;
                // get object score & class score
                cls_stack.clear();
                for (int li = cursor + 5; li < cursor + IN_C; li++) {
                    cls_stack.push_back(input_memory[li]);
                }
                int label = max_element(cls_stack.begin(), cls_stack.end()) - cls_stack.begin();  // 0, 1, 2, ...
                float cls_score = sigmoid(cls_stack[label]);
                float obj_score = sigmoid(input_memory[cursor + 4]);
                if (obj_score * cls_score > CONFIDENCE_THRESH) {
                    // get real-size x, y, w, h, class label, class score
                    float _cx = (sigmoid(input_memory[cursor + 0]) * 2.0 - 0.5 + w) * strides[i];
                    float _cy = (sigmoid(input_memory[cursor + 1]) * 2.0 - 0.5 + h) * strides[i];
                    float _cw = pow(sigmoid(input_memory[cursor + 2]) * 2, 2) * anchors[2 * i];
                    float _ch = pow(sigmoid(input_memory[cursor + 3]) * 2, 2) * anchors[2 * i + 1];
                    // Box (center x, center y, width, height) to (x1, y1, x2, y2)
                    BoundingBox obj;
                    obj.x1 = clip((_cx - _cw / 2.0) / INPUT_WIDTH, 0, 1);
                    obj.y1 = clip((_cy - _ch / 2.0) / INPUT_HEIGHT, 0, 1);
                    obj.x2 = clip((_cx + _cw / 2.0) / INPUT_WIDTH, 0, 1);
                    obj.y2 = clip((_cy + _ch / 2.0) / INPUT_HEIGHT, 0, 1);
                    obj.score = obj_score;
                    obj.label = label;
                    results.push_back(obj);
                    // printf("> %d, %f, %f, %f, %f, %f \n", obj.label, obj.x1, obj.y1, obj.x2, obj.y2, obj.score);
                }
            }
        }
    }
    // run nms
    detection_nms(results);

    // memset top data
    for (int i = 0; i < MAX_DETECTIONS * 6; i++) {
        output_memory[i] = 0.0f;
    }
    int voc_map_labels[] = {1, 2, 3};  // background-0, person-1, bike-2, car-3
    for (int i = 0; i < results.size(); i++) {
        // printf("[%d], %d, %f, %f, %f, %f, %f \n", i, voc_map_labels[results[i].label], results[i].score,
        // results[i].x1,
        //        results[i].y1, results[i].x2, results[i].y2);
        output_memory[i * 6 + 0] = voc_map_labels[results[i].label];
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
        top_shape[2] = 6;
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
        descr.addInputLayout(MEMORY_LAYOUT_XI_WHDN);
        descr.addInputLayout(MEMORY_LAYOUT_XI_WHDN);
        descr.addInputLayout(MEMORY_LAYOUT_XI_WHDN);
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