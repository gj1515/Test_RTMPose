// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/pose_tracker.hpp"
#include "utils/argparse.h"
#include "utils/mediaio.h"
#include "utils/visualize.h"
#include "pose_tracker_params.h"


int main(int argc, char* argv[]) {
    std::string det_model = "D:/Dev/Project/Test_RTMPose/models/RTMDet/rtmdet-n_320x320";
    std::string pose_model = "D:/Dev/Project/Test_RTMPose/models/RTMPose/cocowholebody_rtmpose_l_256x192";
    std::string input = "C:/Capture/0408_Calibration/HSH/motion1/export/Teli_3.mp4";
    std::string output = "D:/Dev/Project/Test_RTMPose/demo/outputs/output.mp4";

    std::string skeleton = "coco-wholebody";  // "coco", "coco-wholebody", "halpe26"
    std::string device = "cuda";              // "cuda", "cpu"
    int output_size = 0;                      // "Long-edge of output frames" (0: original size)
    int flip = 0;                             // "Set to 1 for flipping the input horizontally"
    int show = -1;                            // "Delay passed to `cv::waitKey` when using `cv::imshow`; -1: disable"
    std::string background = "default";       // Output background, "default": original image, "black": black background


    // create pose tracker pipeline
    mmdeploy::PoseTracker tracker(mmdeploy::Model(det_model), mmdeploy::Model(pose_model), mmdeploy::Device{ device });

    mmdeploy::PoseTracker::Params params;
    // initialize tracker params with program arguments
    InitTrackerParams(params);

    // create a tracker state for each video
    mmdeploy::PoseTracker::State state = tracker.CreateState(params);

    utils::mediaio::Input input_stream(input, flip);
    double input_fps = input_stream.get_fps();
    utils::mediaio::Output output_stream(output, show, input_fps);

    utils::Visualize v(output_size);
    v.set_background(background);
    v.set_skeleton(utils::Skeleton::get(skeleton));

    for (const cv::Mat& frame : input_stream) {
        // apply the pipeline with the tracker state and video frame; the result is an array-like class
        // holding references to `mmdeploy_pose_tracker_target_t`, will be released automatically on
        // destruction
        mmdeploy::PoseTracker::Result result = tracker.Apply(state, frame);

        // visualize results
        auto sess = v.get_session(frame);
        for (const mmdeploy_pose_tracker_target_t& target : result) {
            sess.add_pose(target.keypoints, target.scores, target.keypoint_count, FLAGS_pose_kpt_thr);
        }

        // write to output stream
        if (!output_stream.write(sess.get())) {
            // user requested exit by pressing ESC
            break;
        }
    }

    return 0;
}