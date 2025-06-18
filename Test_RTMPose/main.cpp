// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/pose_tracker.hpp"
#include "utils/argparse.h"
#include "utils/mediaio.h"
#include "utils/visualize.h"
#include "pose_tracker_params.h"

#include <chrono>
#include <iostream>

int main(int argc, char* argv[]) {
    std::string det_model = "D:/Dev/Project/mmdeploy/rtmpose-ort/rtmdet-nano";
    std::string pose_model = "C:/Users/user/Downloads/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.zip";
    std::string input = "D:/Dev/Dataset/inputs/videos/Teli_3.mp4";
    std::string output = "D:/Dev/Project/Test_RTMPose/demo/outputs/output.mp4";

    std::string device = "cuda";                             // "Device name, e.g. \"cpu\", \"cuda\""
    int output_size = 0;                                     // "Long-edge of output frames" (0: original size)
    int flip = 0;                                            // "Set to 1 for flipping the input horizontally"
    int show = -1;                                           // "Delay passed to `cv::waitKey` when using `cv::imshow`; -1: disable"
    std::string skeleton = "halpe26";                        // Path to skeleton data or name of predefined skeletons: "coco", "coco-wholebody", "coco-wholebody-hand", "halpe26"
    std::string background = "default";                      // Output background, "default": original image, "black": black background

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

    // *************************************************************
    auto program_start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    auto last_fps_time = std::chrono::high_resolution_clock::now();
    int fps_frame_count = 0;
    // *************************************************************


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

        // *************************************************************
        frame_count++;
        fps_frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed_since_last_fps = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_fps_time);
        // *************************************************************
    }

    // *************************************************************
    auto program_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(program_end - program_start);
    double average_fps = frame_count * 1000.0 / total_time.count();
    std::cout << "\n=== FPS Statistics ===" << std::endl;
    std::cout << "Total frames processed: " << frame_count << std::endl;
    std::cout << "Total processing time: " << total_time.count() << " ms" << std::endl;
    std::cout << "Average FPS: " << std::fixed << std::setprecision(2) << average_fps << std::endl;
    // *************************************************************

    return 0;
}
