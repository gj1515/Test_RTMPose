// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/detector.hpp"
#include "mmdeploy/pose_detector.hpp"
#include "utils/argparse.h"
#include "utils/mediaio.h"
#include "utils/visualize.h"


int main(int argc, char* argv[]) {
    std::string det_model = "D:/Dev/Project/Test_RTMPose/models/RTMDet/rtmdet-m_640x640";
    std::string pose_model = "D:/Dev/Project/Test_RTMPose/models/RTMPose/coco17/384x288/rtmpose_coco17_l_384x288";
    std::string input = "D:/Dev/Dataset/inputs/videos/Teli_3.mp4";
    std::string output = "D:/Dev/Project/Test_RTMPose/demo/outputs/Teli_3_coco17.mp4";

    std::string skeleton = "coco";  // "coco", "coco-wholebody", "halpe26"
    std::string device = "cuda";              // "cuda", "cpu"
    int output_size = 0;                      // "Long-edge of output frames" (0: original size)
    int flip = 0;                             // "Set to 1 for flipping the input horizontally"
    int show = -1;                            // "Delay passed to `cv::waitKey` when using `cv::imshow`; -1: disable"
    std::string background = "default";       // Output background, "default": original image, "black": black background

    int det_label = 0;
    double det_thr = 0.5;
    double pose_kpt_thr = 0.5;

    mmdeploy::Device rtmpose_device{ device };
    // create object detector
    mmdeploy::Detector detector(mmdeploy::Model(det_model), rtmpose_device);
    // create pose detector
    mmdeploy::PoseDetector pose(mmdeploy::Model(pose_model), rtmpose_device);

    utils::mediaio::Input input_stream(input, flip);
    double input_fps = input_stream.get_fps();
    utils::mediaio::Output output_stream(output, show, input_fps);

    utils::Visualize v(output_size);
    v.set_background(background);
    v.set_skeleton(utils::Skeleton::get(skeleton));

    for (const cv::Mat& frame : input_stream) {
        mmdeploy::Detector::Result detections = detector.Apply(frame);

        std::vector<mmdeploy_rect_t> bboxes;
        for (const mmdeploy_detection_t& det : detections) {
            if (det.label_id == det_label && det.score > det_thr) {
                bboxes.push_back(det.bbox);
            }
        }

        mmdeploy::PoseDetector::Result poses = pose.Apply(frame, bboxes);

        // Postprocess & Visualization
        auto sess = v.get_session(frame);
        for (size_t i = 0; i < poses.size(); ++i) {
            sess.add_pose(poses[i].point, poses[i].score, poses[i].length, pose_kpt_thr);
        }

        // write to output stream
        if (!output_stream.write(sess.get())) {
            // user requested exit by pressing ESC
            break;
        }
    }
    return 0;
}
